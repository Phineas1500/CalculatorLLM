#!/usr/bin/env python3
"""
Debug script that exactly matches the C integer implementation.
This should produce the SAME output as the calculator.
"""
import numpy as np
import struct

# Constants - must match gru.c
VOCAB_SIZE = 95
HIDDEN_DIM = 48
EMBED_DIM = 48
GATE_SHIFT = 8
BIAS_SCALE = 128

# LUT tables - must match lut.h exactly
sigmoid_lut_q15 = np.array([
       10,    11,    12,    13,    14,    15,    15,    17,
       18,    19,    20,    21,    23,    24,    26,    28,
       29,    31,    33,    36,    38,    40,    43,    46,
       49,    52,    55,    59,    63,    67,    71,    76,
       81,    86,    91,    97,   103,   110,   117,   125,
      133,   141,   151,   160,   171,   182,   193,   206,
      219,   233,   248,   264,   281,   299,   318,   338,
      360,   382,   407,   433,   460,   490,   521,   554,
      589,   626,   666,   708,   752,   800,   850,   903,
      960,  1020,  1084,  1151,  1223,  1298,  1379,  1464,
     1554,  1649,  1749,  1856,  1968,  2087,  2213,  2345,
     2485,  2633,  2788,  2952,  3124,  3305,  3495,  3695,
     3905,  4126,  4356,  4598,  4851,  5115,  5390,  5678,
     5977,  6289,  6612,  6948,  7297,  7657,  8030,  8415,
     8812,  9220,  9640, 10071, 10512, 10963, 11424, 11893,
    12370, 12855, 13347, 13844, 14346, 14852, 15360, 15871,
    16383, 16895, 17406, 17914, 18420, 18922, 19419, 19911,
    20396, 20873, 21342, 21803, 22254, 22695, 23126, 23546,
    23954, 24351, 24736, 25109, 25469, 25818, 26154, 26477,
    26789, 27088, 27376, 27651, 27915, 28168, 28410, 28640,
    28861, 29071, 29271, 29461, 29642, 29814, 29978, 30133,
    30281, 30421, 30553, 30679, 30798, 30910, 31017, 31117,
    31212, 31302, 31387, 31468, 31543, 31615, 31682, 31746,
    31806, 31863, 31916, 31966, 32014, 32058, 32100, 32140,
    32177, 32212, 32245, 32276, 32306, 32333, 32359, 32384,
    32406, 32428, 32448, 32467, 32485, 32502, 32518, 32533,
    32547, 32560, 32573, 32584, 32595, 32606, 32615, 32625,
    32633, 32641, 32649, 32656, 32663, 32669, 32675, 32680,
    32685, 32690, 32695, 32699, 32703, 32707, 32711, 32714,
    32717, 32720, 32723, 32726, 32728, 32730, 32733, 32735,
    32737, 32738, 32740, 32742, 32743, 32745, 32746, 32747,
    32748, 32749, 32751, 32751, 32752, 32753, 32754, 32755
], dtype=np.int16)

tanh_lut_q15 = np.array([
   -32745,-32743,-32742,-32740,-32738,-32736,-32735,-32732,
   -32730,-32728,-32725,-32723,-32720,-32717,-32714,-32710,
   -32707,-32703,-32699,-32694,-32690,-32685,-32680,-32674,
   -32668,-32662,-32655,-32648,-32640,-32632,-32623,-32614,
   -32604,-32594,-32583,-32571,-32559,-32545,-32531,-32516,
   -32500,-32483,-32464,-32445,-32424,-32402,-32379,-32354,
   -32328,-32300,-32270,-32238,-32204,-32168,-32130,-32090,
   -32046,-32001,-31952,-31900,-31845,-31786,-31724,-31658,
   -31588,-31513,-31434,-31350,-31261,-31166,-31065,-30959,
   -30846,-30726,-30598,-30463,-30320,-30169,-30008,-29838,
   -29658,-29468,-29267,-29054,-28829,-28591,-28340,-28075,
   -27795,-27500,-27190,-26862,-26518,-26156,-25775,-25375,
   -24955,-24514,-24053,-23570,-23064,-22536,-21985,-21410,
   -20811,-20188,-19541,-18869,-18172,-17451,-16705,-15935,
   -15142,-14325,-13485,-12624,-11742,-10840, -9918, -8980,
    -8025, -7055, -6072, -5078, -4074, -3062, -2045, -1023,
        0,  1023,  2045,  3062,  4074,  5078,  6072,  7055,
     8025,  8980,  9918, 10840, 11742, 12624, 13485, 14325,
    15142, 15935, 16705, 17451, 18172, 18869, 19541, 20188,
    20811, 21410, 21985, 22536, 23064, 23570, 24053, 24514,
    24955, 25375, 25775, 26156, 26518, 26862, 27190, 27500,
    27795, 28075, 28340, 28591, 28829, 29054, 29267, 29468,
    29658, 29838, 30008, 30169, 30320, 30463, 30598, 30726,
    30846, 30959, 31065, 31166, 31261, 31350, 31434, 31513,
    31588, 31658, 31724, 31786, 31845, 31900, 31952, 32001,
    32046, 32090, 32130, 32168, 32204, 32238, 32270, 32300,
    32328, 32354, 32379, 32402, 32424, 32445, 32464, 32483,
    32500, 32516, 32531, 32545, 32559, 32571, 32583, 32594,
    32604, 32614, 32623, 32632, 32640, 32648, 32655, 32662,
    32668, 32674, 32680, 32685, 32690, 32694, 32699, 32703,
    32707, 32710, 32714, 32717, 32720, 32723, 32725, 32728,
    32730, 32732, 32735, 32736, 32738, 32740, 32742, 32743
], dtype=np.int16)


def load_weights(path='model_weights.bin'):
    """Load weights exactly as C code does."""
    with open(path, 'rb') as f:
        data = f.read()
    
    # Skip first 4 bytes (float scale)
    scale = struct.unpack('<f', data[:4])[0]
    print(f"Scale from file: {scale}")
    
    # Rest is int8 weights
    weights = np.frombuffer(data[4:], dtype=np.int8)
    print(f"Total weights: {len(weights)} bytes")
    
    # Parse in same order as C
    ptr = 0
    
    embed = weights[ptr:ptr + VOCAB_SIZE * EMBED_DIM].reshape(VOCAB_SIZE, EMBED_DIM)
    ptr += VOCAB_SIZE * EMBED_DIM
    
    W_ir = weights[ptr:ptr + HIDDEN_DIM * EMBED_DIM].reshape(HIDDEN_DIM, EMBED_DIM)
    ptr += HIDDEN_DIM * EMBED_DIM
    
    W_hr = weights[ptr:ptr + HIDDEN_DIM * HIDDEN_DIM].reshape(HIDDEN_DIM, HIDDEN_DIM)
    ptr += HIDDEN_DIM * HIDDEN_DIM
    
    b_ir = weights[ptr:ptr + HIDDEN_DIM]
    ptr += HIDDEN_DIM
    
    b_hr = weights[ptr:ptr + HIDDEN_DIM]
    ptr += HIDDEN_DIM
    
    W_iz = weights[ptr:ptr + HIDDEN_DIM * EMBED_DIM].reshape(HIDDEN_DIM, EMBED_DIM)
    ptr += HIDDEN_DIM * EMBED_DIM
    
    W_hz = weights[ptr:ptr + HIDDEN_DIM * HIDDEN_DIM].reshape(HIDDEN_DIM, HIDDEN_DIM)
    ptr += HIDDEN_DIM * HIDDEN_DIM
    
    b_iz = weights[ptr:ptr + HIDDEN_DIM]
    ptr += HIDDEN_DIM
    
    b_hz = weights[ptr:ptr + HIDDEN_DIM]
    ptr += HIDDEN_DIM
    
    W_in = weights[ptr:ptr + HIDDEN_DIM * EMBED_DIM].reshape(HIDDEN_DIM, EMBED_DIM)
    ptr += HIDDEN_DIM * EMBED_DIM
    
    W_hn = weights[ptr:ptr + HIDDEN_DIM * HIDDEN_DIM].reshape(HIDDEN_DIM, HIDDEN_DIM)
    ptr += HIDDEN_DIM * HIDDEN_DIM
    
    b_in = weights[ptr:ptr + HIDDEN_DIM]
    ptr += HIDDEN_DIM
    
    b_hn = weights[ptr:ptr + HIDDEN_DIM]
    ptr += HIDDEN_DIM
    
    W_out = weights[ptr:ptr + VOCAB_SIZE * HIDDEN_DIM].reshape(VOCAB_SIZE, HIDDEN_DIM)
    ptr += VOCAB_SIZE * HIDDEN_DIM
    
    b_out = weights[ptr:ptr + VOCAB_SIZE]
    ptr += VOCAB_SIZE
    
    print(f"Parsed {ptr} weight bytes")
    
    return {
        'embed': embed, 
        'W_ir': W_ir, 'W_hr': W_hr, 'b_ir': b_ir, 'b_hr': b_hr,
        'W_iz': W_iz, 'W_hz': W_hz, 'b_iz': b_iz, 'b_hz': b_hz,
        'W_in': W_in, 'W_hn': W_hn, 'b_in': b_in, 'b_hn': b_hn,
        'W_out': W_out, 'b_out': b_out,
        'scale': scale
    }


def gru_forward(w, hidden, input_char, debug=False):
    """
    GRU forward pass - exact match to C implementation.
    Uses int32 for accumulators, int8 for weights/activations.
    """
    # Get embedding
    x = w['embed'][input_char].astype(np.int32)
    
    if debug:
        print(f"\n=== Input char {input_char} ('{chr(input_char+32)}') ===")
        print(f"Embedding x[:5] = {x[:5]}")
        print(f"Hidden h[:5] = {hidden[:5]}")
    
    # Reset gate r
    r = np.zeros(HIDDEN_DIM, dtype=np.int8)
    for i in range(HIDDEN_DIM):
        # Exactly as C: sum = ((int32_t)model->b_ir[i] + model->b_hr[i]) * BIAS_SCALE
        bias_sum = (int(w['b_ir'][i]) + int(w['b_hr'][i])) * BIAS_SCALE
        
        # W_ir @ x
        wx = 0
        for j in range(EMBED_DIM):
            wx += int(w['W_ir'][i, j]) * int(x[j])
        
        # W_hr @ h
        wh = 0
        for j in range(HIDDEN_DIM):
            wh += int(w['W_hr'][i, j]) * int(hidden[j])
        
        total = bias_sum + wx + wh
        
        idx = (total >> GATE_SHIFT) + 128
        idx = max(0, min(255, idx))
        
        r[i] = np.int8(sigmoid_lut_q15[idx] >> 8)
        
        if debug and i < 3:
            print(f"r[{i}]: bias={bias_sum}, wx={wx}, wh={wh}, total={total}, idx={idx}, r={r[i]}")
    
    # Update gate z
    z = np.zeros(HIDDEN_DIM, dtype=np.int8)
    for i in range(HIDDEN_DIM):
        bias_sum = (int(w['b_iz'][i]) + int(w['b_hz'][i])) * BIAS_SCALE
        
        wx = 0
        for j in range(EMBED_DIM):
            wx += int(w['W_iz'][i, j]) * int(x[j])
        
        wh = 0
        for j in range(HIDDEN_DIM):
            wh += int(w['W_hz'][i, j]) * int(hidden[j])
        
        total = bias_sum + wx + wh
        
        idx = (total >> GATE_SHIFT) + 128
        idx = max(0, min(255, idx))
        
        z[i] = np.int8(sigmoid_lut_q15[idx] >> 8)
        
        if debug and i < 3:
            print(f"z[{i}]: bias={bias_sum}, wx={wx}, wh={wh}, total={total}, idx={idx}, z={z[i]}")
    
    # Candidate n (with reset gate applied)
    n = np.zeros(HIDDEN_DIM, dtype=np.int8)
    for i in range(HIDDEN_DIM):
        # First part: b_in * BIAS_SCALE + W_in @ x
        inp_sum = int(w['b_in'][i]) * BIAS_SCALE
        for j in range(EMBED_DIM):
            inp_sum += int(w['W_in'][i, j]) * int(x[j])
        
        # Hidden part: b_hn * BIAS_SCALE + W_hn @ h
        h_sum = int(w['b_hn'][i]) * BIAS_SCALE
        for j in range(HIDDEN_DIM):
            h_sum += int(w['W_hn'][i, j]) * int(hidden[j])
        
        # Apply reset gate: r * h_sum >> 7
        gated = (int(r[i]) * h_sum) >> 7
        
        total = inp_sum + gated
        
        idx = (total >> GATE_SHIFT) + 128
        idx = max(0, min(255, idx))
        
        n[i] = np.int8(tanh_lut_q15[idx] >> 8)
        
        if debug and i < 3:
            print(f"n[{i}]: inp_sum={inp_sum}, h_sum={h_sum}, r[i]={r[i]}, gated={gated}, total={total}, idx={idx}, n={n[i]}")
    
    # Update hidden state: h' = (1-z)*n + z*h
    new_hidden = np.zeros(HIDDEN_DIM, dtype=np.int8)
    for i in range(HIDDEN_DIM):
        # C code: ((127 - z[i]) * n[i] + z[i] * model->hidden[i]) >> 7
        val = ((127 - int(z[i])) * int(n[i]) + int(z[i]) * int(hidden[i])) >> 7
        val = max(-128, min(127, val))
        new_hidden[i] = np.int8(val)
        
        if debug and i < 3:
            print(f"h'[{i}]: (127-{z[i]})*{n[i]} + {z[i]}*{hidden[i]} >> 7 = {val}")
    
    if debug:
        print(f"New hidden h[:5] = {new_hidden[:5]}")
    
    # Output logits
    output = np.zeros(VOCAB_SIZE, dtype=np.int32)
    for i in range(VOCAB_SIZE):
        out_sum = int(w['b_out'][i]) * BIAS_SCALE
        for j in range(HIDDEN_DIM):
            out_sum += int(w['W_out'][i, j]) * int(new_hidden[j])
        output[i] = out_sum
    
    if debug:
        top5_idx = np.argsort(output)[-5:][::-1]
        print(f"Top 5 output indices: {top5_idx}")
        for idx in top5_idx:
            print(f"  output[{idx}] ('{chr(idx+32)}') = {output[idx]}")
    
    return new_hidden, output


def main():
    print("=== Debug Match C Implementation ===\n")
    
    w = load_weights()
    
    # Verify some weight values
    print(f"\n=== Weight Verification ===")
    print(f"embed[0][:5] = {w['embed'][0][:5]}")
    print(f"b_out argmax = {np.argmax(w['b_out'])} ('{chr(np.argmax(w['b_out'])+32)}')")
    print(f"b_out[26] (':') = {w['b_out'][26]}")
    print(f"b_out[69] ('e') = {w['b_out'][69]}")
    
    # Test with "The " seed (same as calculator test)
    seed = "The "
    print(f"\n=== Processing seed: '{seed}' ===")
    
    hidden = np.zeros(HIDDEN_DIM, dtype=np.int8)
    
    output = None
    for i, c in enumerate(seed):
        char_idx = ord(c) - 32 if 32 <= ord(c) <= 126 else 0
        print(f"\n--- Character {i}: '{c}' (idx={char_idx}) ---")
        hidden, output = gru_forward(w, hidden, char_idx, debug=(i == len(seed)-1))
    
    # Final prediction
    assert output is not None, "No seed characters processed"
    next_idx = np.argmax(output)
    next_char = chr(next_idx + 32)
    print(f"\n=== RESULT ===")
    print(f"Next character index: {next_idx}")
    print(f"Next character: '{next_char}'")
    print(f"Output logit: {output[next_idx]}")
    
    # Generate a few more
    print(f"\n=== Generating 10 characters ===")
    generated = ""
    for step in range(10):
        hidden, output = gru_forward(w, hidden, next_idx, debug=False)
        next_idx = np.argmax(output)
        next_char = chr(next_idx + 32)
        generated += next_char
        print(f"Step {step}: '{next_char}' (idx={next_idx})")
    
    print(f"\nFull generation: '{seed}{generated}'")


if __name__ == '__main__':
    main()
