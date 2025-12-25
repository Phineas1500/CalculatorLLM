#ifndef MODEL_IO_H
#define MODEL_IO_H

#include "gru.h"
#include <stdbool.h>
#include <stdint.h>

/*
 * Model I/O for TI-84 Plus CE
 *
 * Supports two loading modes:
 * 1. Single AppVar (GRUMODEL) for models <= 64KB
 * 2. Split AppVars (GRUMDL1 + GRUMDL2) for larger models
 *
 * Format: [4-byte float scale] [int8 weights...]
 */

/* Single AppVar name (for smaller models) */
#define MODEL_APPVAR_NAME "GRUMODEL"

/* Split AppVar names (for larger models) */
#define MODEL_APPVAR_PART1 "GRUMDL1"
#define MODEL_APPVAR_PART2 "GRUMDL2"
#define MODEL_SPLIT_SIZE 64000 /* Bytes in first AppVar */

#define MODEL_NUM_WEIGHTS                                                      \
  ((VOCAB_SIZE * EMBED_DIM) +  /* embed */                                     \
   (HIDDEN_DIM * EMBED_DIM) +  /* W_ir */                                      \
   (HIDDEN_DIM * HIDDEN_DIM) + /* W_hr */                                      \
   (HIDDEN_DIM) +              /* b_ir */                                      \
   (HIDDEN_DIM) +              /* b_hr */                                      \
   (HIDDEN_DIM * EMBED_DIM) +  /* W_iz */                                      \
   (HIDDEN_DIM * HIDDEN_DIM) + /* W_hz */                                      \
   (HIDDEN_DIM) +              /* b_iz */                                      \
   (HIDDEN_DIM) +              /* b_hz */                                      \
   (HIDDEN_DIM * EMBED_DIM) +  /* W_in */                                      \
   (HIDDEN_DIM * HIDDEN_DIM) + /* W_hn */                                      \
   (HIDDEN_DIM) +              /* b_in */                                      \
   (HIDDEN_DIM) +              /* b_hn */                                      \
   (VOCAB_SIZE * HIDDEN_DIM) + /* W_out */                                     \
   (VOCAB_SIZE)                /* b_out */                                     \
  )

/* 4 bytes for scale (HARDCODED, not sizeof(float) which is 6 on ez80) */
#define MODEL_TOTAL_SIZE (4 + MODEL_NUM_WEIGHTS)

/* Check if model needs split loading */
#define MODEL_NEEDS_SPLIT (MODEL_TOTAL_SIZE > 64000)

bool model_load(uint8_t *buffer);
bool model_exists(void);
const uint8_t *model_get_archive_ptr(void);

/* Split loading: returns combined pointer to weights (caller provides buffer)
 */
bool model_load_split(uint8_t *buffer);
bool model_exists_split(void);

#endif /* MODEL_IO_H */
