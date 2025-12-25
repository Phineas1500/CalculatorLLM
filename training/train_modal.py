#!/usr/bin/env python3
"""Train TI-84 GRU model on Modal with GPU acceleration."""

import modal

app = modal.App("ti84-gru-training")

# Create a volume to persist training data and outputs
volume = modal.Volume.from_name("ti84-training-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy",
)

@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
    volumes={"/data": volume},
)
def train_model(corpus_text: str, epochs: int = 50):
    """Train the GRU model on GPU and return weights with per-tensor scaling."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import time

    VOCAB_SIZE = 95
    EMBED_DIM = 128       # Large model, uses split AppVars
    HIDDEN_DIM = 128      # Large model, uses split AppVars
    SEQ_LENGTH = 64

    class CharDataset(Dataset):
        def __init__(self, text: str, seq_length: int = SEQ_LENGTH, stride: int = 1):
            self.seq_length = seq_length
            self.stride = stride
            data = []
            for c in text:
                if 32 <= ord(c) <= 126:
                    data.append(ord(c) - 32)
                else:
                    data.append(0)
            self.data = torch.tensor(data, dtype=torch.long)
            self.num_sequences = max(0, (len(self.data) - seq_length - 1) // stride)

        def __len__(self):
            return self.num_sequences

        def __getitem__(self, idx):
            start = idx * self.stride
            x = self.data[start:start + self.seq_length]
            y = self.data[start + 1:start + self.seq_length + 1]
            return x, y

    class TinyGRU(nn.Module):
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


    # Setup
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataset
    dataset = CharDataset(corpus_text, SEQ_LENGTH, stride=3)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True, num_workers=2)
    print(f"Dataset: {len(dataset):,} sequences")
    print(f"Batches per epoch: {len(dataloader)}")

    # Create model
    model = TinyGRU().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters")

    # Train WITHOUT weight clipping - let model learn freely
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== Training for {epochs} epochs (no clipping) ===\n")
    start_time = time.time()

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
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")

        if (epoch + 1) % 10 == 0:
            sample = model.generate("The ", max_length=50, temperature=0.8)
            print(f"  Sample: {sample}\n")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Check weight range
    all_params = torch.cat([p.data.flatten() for p in model.parameters()])
    print(f"Weight range: [{all_params.min():.3f}, {all_params.max():.3f}]")

    # Export weights as int8 with global scale factor
    model.eval()
    model.cpu()

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

    # List of all tensors in order (must match C code gru_init)
    tensors = [
        ("embed", embed),
        ("W_ir", W_ir), ("W_hr", W_hr), ("b_ir", b_ir), ("b_hr", b_hr),
        ("W_iz", W_iz), ("W_hz", W_hz), ("b_iz", b_iz), ("b_hz", b_hz),
        ("W_in", W_in), ("W_hn", W_hn), ("b_in", b_in), ("b_hn", b_hn),
        ("W_out", W_out), ("b_out", b_out),
    ]

    # Find global scale factor
    all_flat = torch.cat([t.flatten() for _, t in tensors])
    global_scale = all_flat.abs().max().item()
    print(f"\nGlobal scale factor: {global_scale:.4f}")

    # Quantize all weights to int8
    def quantize(t):
        scaled = t / global_scale * 127.0
        return scaled.clamp(-127, 127).round().to(torch.int8).numpy()

    all_weights = []
    print("\n=== Quantized weight tensors ===")
    for name, tensor in tensors:
        q = quantize(tensor).flatten()
        all_weights.append(q)
        print(f"{name}: shape={tensor.shape}, {len(q)} bytes")

    # Pack: 4-byte float scale + int8 weights
    import struct
    scale_bytes = struct.pack('<f', global_scale)
    weight_bytes = scale_bytes + np.concatenate(all_weights).tobytes()
    print(f"\nTotal: {len(weight_bytes)} bytes ({len(weight_bytes)/1024:.1f} KB)")
    print(f"  Scale factor: 4 bytes")
    print(f"  Weights: {len(weight_bytes) - 4} bytes")

    # Generate samples
    model.to('cuda')
    print("\n=== Sample Generation ===")
    for seed in ["The ", "To be ", "I think "]:
        generated = model.generate(seed, max_length=80, temperature=0.7)
        print(f"'{seed}' -> {generated}")

    return weight_bytes


@app.local_entrypoint()
def main():
    from pathlib import Path

    # Load corpus
    corpus_path = Path("corpus.txt")
    if not corpus_path.exists():
        print("Error: corpus.txt not found in current directory")
        return

    corpus_text = corpus_path.read_text()
    print(f"Loaded {len(corpus_text):,} characters from corpus.txt")

    # Train on Modal
    print("\nStarting training on Modal GPU...")
    weights_bytes = train_model.remote(corpus_text, epochs=50)

    # Save weights locally
    output_path = Path("model_weights.bin")
    output_path.write_bytes(weights_bytes)
    print(f"\nSaved weights to {output_path}")

    # Convert to AppVar (using split script for large models)
    print("Converting to AppVar format...")
    import subprocess
    subprocess.run(["python3", "split_weights.py"])
