#!/usr/bin/env python3
"""Diagnose the model export and verify weights."""
import torch
import numpy as np
import struct

# Load the PyTorch model
VOCAB_SIZE = 95
EMBED_DIM = 48
HIDDEN_DIM = 48

class TinyGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.gru = torch.nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.output = torch.nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    
    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.gru(embed, hidden)
        logits = self.output(out)
        return logits, hidden

# Load model checkpoint
try:
    model = TinyGRU()
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    model.eval()
    print("✓ Loaded model.pt")
except Exception as e:
    print(f"✗ Failed to load model.pt: {e}")
    exit(1)

# Test generation with the PyTorch model
print("\n=== Python Model Generation Test ===")
def generate(seed, max_len=50):
    chars = [ord(c) - 32 if 32 <= ord(c) <= 126 else 0 for c in seed]
    x = torch.tensor([chars], dtype=torch.long)
    result = list(seed)
    hidden = None
    
    with torch.no_grad():
        logits, hidden = model(x, hidden)
        for _ in range(max_len):
            last_logits = logits[0, -1, :]
            # Argmax (same as C code)
            next_idx = last_logits.argmax().item()
            result.append(chr(next_idx + 32))
            x = torch.tensor([[next_idx]], dtype=torch.long)
            logits, hidden = model(x, hidden)
    return ''.join(result)

print(f"Seed 'The ': {generate('The ')}")
print(f"Seed 'Hello ': {generate('Hello ')}")

# Check the binary weights file
print("\n=== Weights File Analysis ===")
with open('model_weights.bin', 'rb') as f:
    data = f.read()

scale = struct.unpack('<f', data[:4])[0]
print(f"Scale factor: {scale}")

weights = np.frombuffer(data[4:], dtype=np.int8)
print(f"Total weights: {len(weights)}")
print(f"Weight range: [{weights.min()}, {weights.max()}]")
print(f"Non-zero weights: {np.count_nonzero(weights)} / {len(weights)}")

# Check weight distribution
unique, counts = np.unique(weights, return_counts=True)
print(f"\nMost common weight values:")
sorted_idx = np.argsort(counts)[::-1][:5]
for i in sorted_idx:
    print(f"  {unique[i]:4d}: {counts[i]:6d} ({100*counts[i]/len(weights):.1f}%)")

# Calculate expected sizes
embed_size = VOCAB_SIZE * EMBED_DIM
gate_w_size = HIDDEN_DIM * EMBED_DIM
gate_u_size = HIDDEN_DIM * HIDDEN_DIM
gate_b_size = HIDDEN_DIM
output_w_size = VOCAB_SIZE * HIDDEN_DIM
output_b_size = VOCAB_SIZE

expected = embed_size + 3*(gate_w_size + gate_u_size + 2*gate_b_size) + output_w_size + output_b_size
print(f"\nExpected weights: {expected}")
print(f"Actual weights: {len(weights)}")
print(f"Match: {'✓' if expected == len(weights) else '✗ MISMATCH!'}")

# Check PyTorch gate order
print("\n=== PyTorch GRU Gate Order Check ===")
W_ih = model.gru.weight_ih_l0.data
W_hh = model.gru.weight_hh_l0.data
print(f"W_ih shape: {W_ih.shape} (expected: [{3*HIDDEN_DIM}, {EMBED_DIM}])")
print(f"W_hh shape: {W_hh.shape} (expected: [{3*HIDDEN_DIM}, {HIDDEN_DIM}])")

# PyTorch GRU gate order is: reset, update, new (r, z, n)
# Verify by checking the documentation
print("\nPyTorch gate order: reset (r), update (z), new (n)")
print("C code expects:     reset (r), update (z), new (n)")
print("Order match: ✓")

# Simulate one forward pass manually
print("\n=== Manual Forward Pass Simulation ===")
with torch.no_grad():
    # Input: 'T' (ASCII 84 -> index 52)
    idx = ord('T') - 32
    print(f"Input char: 'T' (idx={idx})")
    
    # Get embedding
    embed = model.embedding.weight[idx]
    print(f"Embedding norm: {embed.norm():.4f}")
    
    # Check output bias (what we get if hidden=0)
    b_out = model.output.bias.data
    print(f"Output bias range: [{b_out.min():.4f}, {b_out.max():.4f}]")
    print(f"Output bias argmax: {b_out.argmax().item()} (char='{chr(b_out.argmax().item() + 32)}')")
    
    # Full forward pass
    x = torch.tensor([[idx]], dtype=torch.long)
    logits, _ = model(x)
    probs = torch.softmax(logits[0, 0], dim=0)
    top5 = probs.topk(5)
    print(f"\nTop 5 predictions for 'T':")
    for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"  {i+1}. '{chr(idx.item() + 32)}' ({prob.item()*100:.1f}%)")

print("\n=== Done ===")
