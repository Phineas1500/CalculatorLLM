#!/usr/bin/env python3
"""
Training script for TI-84 Plus CE character-level GRU model.

This trains a small GRU on text data (e.g., Tiny Shakespeare) and
exports the weights in Q7 fixed-point format for the calculator.

Usage:
    python train.py --data corpus.txt --epochs 20 --output model_weights.bin
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# Model hyperparameters (must match C code)
VOCAB_SIZE = 95      # Printable ASCII: space (32) to tilde (126)
EMBED_DIM = 128      # Large model, uses split AppVars
HIDDEN_DIM = 128     # Large model, uses split AppVars
SEQ_LENGTH = 64      # Training sequence length


class CharDataset(Dataset):
    """Character-level dataset for text generation."""

    def __init__(self, text: str, seq_length: int = SEQ_LENGTH, stride: int = 1):
        self.seq_length = seq_length
        self.stride = stride

        # Convert text to indices (ASCII 32-126 -> 0-94)
        data = []
        for c in text:
            if 32 <= ord(c) <= 126:
                data.append(ord(c) - 32)
            else:
                data.append(0)  # Map unknown to space
        self.data = torch.tensor(data, dtype=torch.long)

        # Calculate number of sequences with stride
        self.num_sequences = max(0, (len(self.data) - seq_length - 1) // stride)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start:start + self.seq_length]
        y = self.data[start + 1:start + self.seq_length + 1]
        return x, y


class TinyGRU(nn.Module):
    """Minimal GRU model for character-level text generation."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.gru(embed, hidden)
        logits = self.output(out)
        return logits, hidden

    def generate(self, seed_text: str, max_length: int = 100, temperature: float = 0.8):
        """Generate text from a seed."""
        self.eval()
        device = next(self.parameters()).device

        chars = [ord(c) - 32 if 32 <= ord(c) <= 126 else 0 for c in seed_text]
        x = torch.tensor([chars], dtype=torch.long, device=device)

        generated = list(seed_text)
        hidden = None

        with torch.no_grad():
            logits, hidden = self(x, hidden)

            for _ in range(max_length):
                last_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(last_logits, dim=0)
                next_idx = torch.multinomial(probs, 1).item()

                next_char = chr(next_idx + 32)
                generated.append(next_char)

                x = torch.tensor([[next_idx]], dtype=torch.long, device=device)
                logits, hidden = self(x, hidden)

        return ''.join(generated)


def clip_weights(model, max_val=0.99):
    """Clip all weights to [-max_val, max_val] range."""
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(-max_val, max_val)


def train(model, dataloader, epochs, device, lr=0.001):
    """Train the model with weight clipping for quantization."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)  # L2 regularization
    criterion = nn.CrossEntropyLoss()

    num_batches = len(dataloader)
    print(f"Training: {num_batches} batches per epoch")
    print(f"Using weight decay + clipping for quantization-friendly weights\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)

            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Clip weights to stay in quantization-friendly range
            clip_weights(model, 0.99)

            total_loss += loss.item()

            # Progress update every 10% of epoch
            if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                pct = 100 * (batch_idx + 1) / num_batches
                print(f"\r  Epoch {epoch + 1}: {pct:.0f}% ({batch_idx + 1}/{num_batches})", end="", flush=True)

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        print(f"\r  Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")

        # Generate sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            sample = model.generate("The ", max_length=50, temperature=0.8)
            print(f"  Sample: {sample}\n")


def quantize_to_q7(tensor: torch.Tensor) -> np.ndarray:
    """Quantize float tensor to Q7 format (int8)."""
    # Weights should already be in [-1, 1] due to training with clipping
    scaled = tensor * 127.0
    quantized = scaled.clamp(-127, 127).round().to(torch.int8)
    return quantized.cpu().numpy()


def export_weights(model, output_path: str):
    """Export model weights in Q7 format for the calculator.
    
    Output format: [4-byte float scale] [int8 quantized weights...]
    The C code dequantizes as: real_weight = int8_weight * scale / 127
    """
    model.eval()

    all_params = torch.cat([p.data.flatten() for p in model.parameters()])
    scale = all_params.abs().max().item()
    print(f"Weight range: [{all_params.min():.3f}, {all_params.max():.3f}]")
    print(f"Global scale factor: {scale:.6f}")

    def quantize_with_scale(tensor: torch.Tensor) -> np.ndarray:
        scaled = (tensor / scale) * 127.0
        quantized = scaled.clamp(-127, 127).round().to(torch.int8)
        return quantized.cpu().numpy()

    embed = model.embedding.weight.data
    W_ih = model.gru.weight_ih_l0.data
    W_hh = model.gru.weight_hh_l0.data
    b_ih = model.gru.bias_ih_l0.data
    b_hh = model.gru.bias_hh_l0.data
    W_out = model.output.weight.data
    b_out = model.output.bias.data

    W_ir, W_iz, W_in = W_ih.chunk(3, dim=0)
    W_hr, W_hz, W_hn = W_hh.chunk(3, dim=0)
    b_ir, b_iz, b_in = b_ih.chunk(3, dim=0)
    b_hr, b_hz, b_hn = b_hh.chunk(3, dim=0)

    weights = []

    weights.append(quantize_with_scale(embed))
    print(f"Embedding: {embed.shape}")

    for t in [W_ir, W_hr, b_ir, b_hr]:
        weights.append(quantize_with_scale(t))

    for t in [W_iz, W_hz, b_iz, b_hz]:
        weights.append(quantize_with_scale(t))

    for t in [W_in, W_hn, b_in, b_hn]:
        weights.append(quantize_with_scale(t))

    weights.append(quantize_with_scale(W_out))
    weights.append(quantize_with_scale(b_out))
    print(f"Output: {W_out.shape}")

    all_weights = np.concatenate([w.flatten() for w in weights])
    
    with open(output_path, 'wb') as f:
        f.write(np.array([scale], dtype=np.float32).tobytes())
        f.write(all_weights.tobytes())
    
    total_size = 4 + len(all_weights)
    print(f"Total: {total_size} bytes ({total_size/1024:.1f} KB)")
    print(f"  Scale factor: 4 bytes")
    print(f"  Weights: {len(all_weights)} bytes")
    print(f"Saved to: {output_path}")

    return all_weights


def main():
    parser = argparse.ArgumentParser(description='Train TI-84 GRU model')
    parser.add_argument('--data', type=str, default='corpus.txt',
                        help='Path to training text file')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='Learning rate')
    parser.add_argument('--stride', type=int, default=3,
                        help='Stride for sequence sampling (reduces dataset size)')
    parser.add_argument('--output', type=str, default='model_weights.bin',
                        help='Output weights file')
    parser.add_argument('--checkpoint', type=str, default='model.pt',
                        help='Model checkpoint file')
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return

    text = data_path.read_text()
    print(f"Loaded {len(text):,} characters from {data_path}")

    # Create dataset with stride to reduce size
    dataset = CharDataset(text, SEQ_LENGTH, stride=args.stride)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # Avoid small final batch
    )
    print(f"Dataset: {len(dataset):,} sequences (stride={args.stride})")
    print(f"Batches per epoch: {len(dataloader)}")

    # Create model
    model = TinyGRU()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters ({num_params/1000:.1f}K)\n")

    # Train
    print("=== Training ===\n")
    start_time = time.time()
    train(model, dataloader, args.epochs, device, lr=args.lr)
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)\n")

    # Save checkpoint
    torch.save(model.state_dict(), args.checkpoint)
    print(f"Checkpoint: {args.checkpoint}")

    # Export weights
    print("\n=== Exporting Weights ===\n")
    export_weights(model, args.output)

    # Test generation
    print("\n=== Sample Generation ===\n")
    model.to(device)
    for seed in ["The ", "To be ", "I think "]:
        generated = model.generate(seed, max_length=80, temperature=0.7)
        print(f"'{seed}' -> {generated}\n")


if __name__ == '__main__':
    main()
