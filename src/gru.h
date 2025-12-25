#ifndef GRU_H
#define GRU_H

#include <stdint.h>

#define VOCAB_SIZE 95
#define HIDDEN_DIM 128
#define EMBED_DIM 128

#define PREACT_SHIFT 20

typedef struct {
  float scale;

  const int8_t *embed;

  const int8_t *W_ir;
  const int8_t *W_hr;
  const int8_t *b_ir;
  const int8_t *b_hr;

  const int8_t *W_iz;
  const int8_t *W_hz;
  const int8_t *b_iz;
  const int8_t *b_hz;

  const int8_t *W_in;
  const int8_t *W_hn;
  const int8_t *b_in;
  const int8_t *b_hn;

  const int8_t *W_out;
  const int8_t *b_out;

  int8_t hidden[HIDDEN_DIM];

  int16_t scale_q8;
} GRU_Model;

/* Initialize GRU model.
 * If weights_data is NULL, assumes pointers in 'model' are already set
 * (distributed load). If weights_data is provided, parses it as a single
 * contiguous blob.
 */
void gru_init(GRU_Model *model, const uint8_t *weights_data);
void gru_reset_hidden(GRU_Model *model);
void gru_forward(GRU_Model *model, uint8_t input_char, int32_t *output);

static inline uint8_t char_to_idx(char c) {
  if (c >= 32 && c <= 126)
    return (uint8_t)(c - 32);
  return 0;
}

static inline char idx_to_char(uint8_t idx) {
  if (idx < VOCAB_SIZE)
    return (char)(idx + 32);
  return ' ';
}

uint8_t gru_sample(const int32_t *output);
uint8_t gru_sample_topk(const int32_t *output, uint8_t k, uint16_t rand_seed);

#endif
