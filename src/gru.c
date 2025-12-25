#include "gru.h"
#include "lut.h"
#include <debug.h>
#include <string.h>

static int debug_step = 0;

#define EMBED_SIZE (VOCAB_SIZE * EMBED_DIM)
#define GATE_W_SIZE (HIDDEN_DIM * EMBED_DIM)
#define GATE_U_SIZE (HIDDEN_DIM * HIDDEN_DIM)
#define GATE_B_SIZE (HIDDEN_DIM)
#define OUTPUT_W_SIZE (VOCAB_SIZE * HIDDEN_DIM)
#define OUTPUT_B_SIZE (VOCAB_SIZE)

#define GATE_SHIFT 8
#define BIAS_SCALE 128

void gru_init(GRU_Model *model, const uint8_t *weights_data) {
  const uint8_t *ptr = weights_data;
  /* Skip 4-byte float in weight file - ez80 sizeof(float)=6, but file uses 4 */
  ptr += 4; /* Hardcode 4 bytes, NOT sizeof(float) which is 6 on ez80 */
  model->scale_q8 = 236;

  const int8_t *w = (const int8_t *)ptr;
  model->embed = w;
  w += EMBED_SIZE;
  model->W_ir = w;
  w += GATE_W_SIZE;
  model->W_hr = w;
  w += GATE_U_SIZE;
  model->b_ir = w;
  w += GATE_B_SIZE;
  model->b_hr = w;
  w += GATE_B_SIZE;
  model->W_iz = w;
  w += GATE_W_SIZE;
  model->W_hz = w;
  w += GATE_U_SIZE;
  model->b_iz = w;
  w += GATE_B_SIZE;
  model->b_hz = w;
  w += GATE_B_SIZE;
  model->W_in = w;
  w += GATE_W_SIZE;
  model->W_hn = w;
  w += GATE_U_SIZE;
  model->b_in = w;
  w += GATE_B_SIZE;
  model->b_hn = w;
  w += GATE_B_SIZE;
  model->W_out = w;
  w += OUTPUT_W_SIZE;
  model->b_out = w;
  gru_reset_hidden(model);
}

void gru_reset_hidden(GRU_Model *model) {
  memset(model->hidden, 0, HIDDEN_DIM);
}

void gru_forward(GRU_Model *model, uint8_t input_char, int32_t *output) {
  const int8_t *x;
  int8_t r[HIDDEN_DIM], z[HIDDEN_DIM], n[HIDDEN_DIM];
  uint8_t i, j;
  int32_t sum, idx;

  if (input_char >= VOCAB_SIZE)
    input_char = 0;
  x = &model->embed[input_char * EMBED_DIM];

  if (debug_step < 5) {
    dbg_printf("GRU step %d: input_char=%d\n", debug_step, input_char);
    dbg_printf("  x[0..4]=%d,%d,%d,%d,%d\n", x[0], x[1], x[2], x[3], x[4]);
    dbg_printf("  h[0..4]=%d,%d,%d,%d,%d\n", model->hidden[0], model->hidden[1],
               model->hidden[2], model->hidden[3], model->hidden[4]);
  }

  for (i = 0; i < HIDDEN_DIM; i++) {
    sum = ((int32_t)model->b_ir[i] + (int32_t)model->b_hr[i]) * BIAS_SCALE;
    for (j = 0; j < EMBED_DIM; j++)
      sum += (int32_t)model->W_ir[i * EMBED_DIM + j] * (int32_t)x[j];
    for (j = 0; j < HIDDEN_DIM; j++)
      sum +=
          (int32_t)model->W_hr[i * HIDDEN_DIM + j] * (int32_t)model->hidden[j];
    idx = (sum >> GATE_SHIFT) + 128;
    if (idx < 0)
      idx = 0;
    else if (idx > 255)
      idx = 255;
    r[i] = (int8_t)(sigmoid_lut_q15[idx] >> 8);

    if (debug_step == 3 && i < 3) {
      dbg_printf("  r[%d]: sum=%ld idx=%ld r=%d\n", i, (long)sum, (long)idx,
                 r[i]);
    }
  }

  for (i = 0; i < HIDDEN_DIM; i++) {
    sum = ((int32_t)model->b_iz[i] + (int32_t)model->b_hz[i]) * BIAS_SCALE;
    for (j = 0; j < EMBED_DIM; j++)
      sum += (int32_t)model->W_iz[i * EMBED_DIM + j] * (int32_t)x[j];
    for (j = 0; j < HIDDEN_DIM; j++)
      sum +=
          (int32_t)model->W_hz[i * HIDDEN_DIM + j] * (int32_t)model->hidden[j];
    idx = (sum >> GATE_SHIFT) + 128;
    if (idx < 0)
      idx = 0;
    else if (idx > 255)
      idx = 255;
    z[i] = (int8_t)(sigmoid_lut_q15[idx] >> 8);

    if (debug_step == 3 && i < 3) {
      dbg_printf("  z[%d]: sum=%ld idx=%ld z=%d\n", i, (long)sum, (long)idx,
                 z[i]);
    }
  }

  for (i = 0; i < HIDDEN_DIM; i++) {
    sum = (int32_t)model->b_in[i] * BIAS_SCALE;
    for (j = 0; j < EMBED_DIM; j++)
      sum += (int32_t)model->W_in[i * EMBED_DIM + j] * (int32_t)x[j];

    int32_t h_sum = (int32_t)model->b_hn[i] * BIAS_SCALE;
    for (j = 0; j < HIDDEN_DIM; j++)
      h_sum +=
          (int32_t)model->W_hn[i * HIDDEN_DIM + j] * (int32_t)model->hidden[j];

    sum += ((int32_t)r[i] * h_sum) >> 7;

    idx = (sum >> GATE_SHIFT) + 128;
    if (idx < 0)
      idx = 0;
    else if (idx > 255)
      idx = 255;
    n[i] = (int8_t)(tanh_lut_q15[idx] >> 8);

    if (debug_step == 3 && i < 3) {
      dbg_printf("  n[%d]: sum=%ld idx=%ld n=%d\n", i, (long)sum, (long)idx,
                 n[i]);
    }
  }

  for (i = 0; i < HIDDEN_DIM; i++) {
    /* Use int32_t to avoid overflow on ez80 (int is 16-bit) */
    int32_t new_h = ((int32_t)(127 - z[i]) * (int32_t)n[i] +
                     (int32_t)z[i] * (int32_t)model->hidden[i]) >>
                    7;
    if (new_h > 127)
      new_h = 127;
    if (new_h < -128)
      new_h = -128;
    model->hidden[i] = (int8_t)new_h;
  }

  for (i = 0; i < VOCAB_SIZE; i++) {
    sum = (int32_t)model->b_out[i] * BIAS_SCALE;
    for (j = 0; j < HIDDEN_DIM; j++)
      sum +=
          (int32_t)model->W_out[i * HIDDEN_DIM + j] * (int32_t)model->hidden[j];
    output[i] = sum;
  }

  if (debug_step < 5) {
    int32_t max_val = output[0];
    int max_idx = 0;
    for (i = 1; i < VOCAB_SIZE; i++) {
      if (output[i] > max_val) {
        max_val = output[i];
        max_idx = i;
      }
    }
    dbg_printf("  output[0]=' '=%ld, output[69]='e'=%ld\n", (long)output[0],
               (long)output[69]);
    dbg_printf("  argmax=%d ('%c') val=%ld\n", max_idx, (char)(max_idx + 32),
               (long)max_val);
    dbg_printf("  new_h[0..4]=%d,%d,%d,%d,%d\n", model->hidden[0],
               model->hidden[1], model->hidden[2], model->hidden[3],
               model->hidden[4]);
  }
  debug_step++;
}

uint8_t gru_sample(const int32_t *output) {
  uint8_t i, max_idx = 0;
  int32_t max_val = output[0];
  for (i = 1; i < VOCAB_SIZE; i++) {
    if (output[i] > max_val) {
      max_val = output[i];
      max_idx = i;
    }
  }
  return max_idx;
}

uint8_t gru_sample_topk(const int32_t *output, uint8_t k, uint16_t rand_seed) {
  /* Find top-k candidates and randomly sample from them */
  uint8_t top_idx[8]; /* Max k=8 */
  int32_t top_val[8];
  uint8_t i, j, num_top = 0;

  if (k > 8)
    k = 8;
  if (k < 1)
    k = 1;

  /* Initialize with minimum values */
  for (i = 0; i < k; i++) {
    top_val[i] = -2147483647;
    top_idx[i] = 0;
  }

  /* Find top-k values */
  for (i = 0; i < VOCAB_SIZE; i++) {
    /* Find position to insert */
    for (j = 0; j < k; j++) {
      if (output[i] > top_val[j]) {
        /* Shift down */
        uint8_t m;
        for (m = k - 1; m > j; m--) {
          top_val[m] = top_val[m - 1];
          top_idx[m] = top_idx[m - 1];
        }
        top_val[j] = output[i];
        top_idx[j] = i;
        break;
      }
    }
  }

  /* Simple weighted random selection from top-k */
  /* Use difference from top as inverse weight */
  int32_t max_val = top_val[0];
  uint16_t weights[8];
  uint16_t total_weight = 0;

  for (i = 0; i < k; i++) {
    /* Weight = scaled difference from max (closer = higher weight) */
    int32_t diff = max_val - top_val[i];
    /* Use exponential-ish decay: weight = max(1, 256 - diff/16) */
    if (diff < 0)
      diff = 0;
    int32_t w = 256 - (diff >> 4);
    if (w < 1)
      w = 1;
    if (w > 256)
      w = 256;
    weights[i] = (uint16_t)w;
    total_weight += weights[i];
  }

  /* Random selection using seed */
  uint16_t r = (rand_seed * 1103515245 + 12345) & 0x7FFF;
  r = r % total_weight;

  uint16_t cumulative = 0;
  for (i = 0; i < k; i++) {
    cumulative += weights[i];
    if (r < cumulative) {
      return top_idx[i];
    }
  }

  return top_idx[0]; /* Fallback */
}
