#!/usr/bin/env python3
"""Diagnose the trained model and test quantized inference."""

import torch
import numpy as np
from pathlib import Path

# Load the model
from train import TinyGRU, VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM

def analyze_weights(model):
    """Print weight statistics."""
    print("=== Weight Statistics ===\n")

    all_weights = []

    # Embedding
    w = model.embedding.weight.data
    print(f"Embedding [{w.shape}]: min={w.min():.3f}, max={w.max():.3f}, absmax={w.abs().max():.3f}")
    all_weights.append(w.flatten())

    # GRU weights
    for name in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']:
        w = getattr(model.gru, name).data
        print(f"GRU {name} [{w.shape}]: min={w.min():.3f}, max={w.max():.3f}, absmax={w.abs().max():.3f}")
        all_weights.append(w.flatten())

    # Output
    w = model.output.weight.data
    print(f"Output weight [{w.shape}]: min={w.min():.3f}, max={w.max():.3f}, absmax={w.abs().max():.3f}")
    all_weights.append(w.flatten())

    w = model.output.bias.data
    print(f"Output bias [{w.shape}]: min={w.min():.3f}, max={w.max():.3f}, absmax={w.abs().max():.3f}")
    all_weights.append(w.flatten())

    # Global stats
    all_w = torch.cat(all_weights)
    print(f"\nGlobal: min={all_w.min():.3f}, max={all_w.max():.3f}, absmax={all_w.abs().max():.3f}")

    return all_w.abs().max().item()


def test_float_inference(model, seed="The "):
    """Test float model inference."""
    print(f"\n=== Float Inference ===")
    print(f"Seed: '{seed}'")

    device = next(model.parameters()).device
    model.eval()

    # Convert seed to indices
    chars = [ord(c) - 32 if 32 <= ord(c) <= 126 else 0 for c in seed]
    x = torch.tensor([chars], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, hidden = model(x)

        # Show top predictions for next char
        probs = torch.softmax(logits[0, -1, :], dim=0)
        top5 = torch.topk(probs, 5)

        print("Top 5 next character predictions:")
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            char = chr(idx.item() + 32)
            print(f"  {i+1}. '{char}' ({prob.item()*100:.1f}%)")

    # Generate
    generated = model.generate(seed, max_length=50, temperature=0.7)
    print(f"\nGenerated: {generated}")


def test_quantized_step(model, seed="The "):
    """Test a single quantized forward pass."""
    print(f"\n=== Quantized Step Test ===")

    model.eval()
    device = 'cpu'
    model.to(device)

    # Get weights and quantize
    embed = model.embedding.weight.data
    W_ih = model.gru.weight_ih_l0.data
    W_hh = model.gru.weight_hh_l0.data
    b_ih = model.gru.bias_ih_l0.data
    b_hh = model.gru.bias_hh_l0.data
    W_out = model.output.weight.data
    b_out = model.output.bias.data

    # Find global scale
    all_weights = torch.cat([
        embed.flatten(), W_ih.flatten(), W_hh.flatten(),
        b_ih.flatten(), b_hh.flatten(),
        W_out.flatten(), b_out.flatten()
    ])
    global_scale = all_weights.abs().max().item()
    print(f"Global weight scale: {global_scale:.3f}")

    # Quantize with proper scaling
    def quantize(t, scale):
        return (t / scale * 127).clamp(-127, 127).round().to(torch.int8)

    embed_q = quantize(embed, global_scale)
    W_ih_q = quantize(W_ih, global_scale)
    W_hh_q = quantize(W_hh, global_scale)
    b_ih_q = quantize(b_ih, global_scale)
    b_hh_q = quantize(b_hh, global_scale)
    W_out_q = quantize(W_out, global_scale)
    b_out_q = quantize(b_out, global_scale)

    # Test embedding lookup
    char_idx = ord(seed[0]) - 32
    embed_float = embed[char_idx]
    embed_quant = embed_q[char_idx].float() * global_scale / 127

    print(f"\nFirst char '{seed[0]}' (idx={char_idx})")
    print(f"Embedding (float): {embed_float[:5].tolist()}")
    print(f"Embedding (quant): {embed_quant[:5].tolist()}")
    print(f"Embedding error: {(embed_float - embed_quant).abs().mean():.4f}")


def main():
    # Load checkpoint
    checkpoint_path = Path('model.pt')
    if not checkpoint_path.exists():
        print("Error: model.pt not found. Run train.py first.")
        return

    model = TinyGRU()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    # Analyze
    global_scale = analyze_weights(model)

    # Test float
    test_float_inference(model, "The ")
    test_float_inference(model, "Hello ")

    # Test quantized
    test_quantized_step(model, "The ")

    print(f"\n=== Recommendation ===")
    if global_scale > 1.0:
        print(f"Weights exceed [-1,1] range (max={global_scale:.2f})")
        print(f"Need to scale weights by {global_scale:.2f} before quantizing")
        print(f"And scale activations accordingly in C code")
    else:
        print(f"Weights are in [-1,1] range, quantization should work")


if __name__ == '__main__':
    main()
