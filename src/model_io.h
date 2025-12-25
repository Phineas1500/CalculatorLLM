#ifndef MODEL_IO_H
#define MODEL_IO_H

#include <stdint.h>
#include <stdbool.h>
#include "gru.h"

/*
 * Model I/O for TI-84 Plus CE
 *
 * Format: [4-byte float scale] [int8 weights...]
 * Total size: 4 + 23,327 = 23,331 bytes
 */

#define MODEL_APPVAR_NAME "GRUMODEL"

#define MODEL_NUM_WEIGHTS ( \
    (VOCAB_SIZE * EMBED_DIM) +       /* embed */ \
    (HIDDEN_DIM * EMBED_DIM) +       /* W_ir */ \
    (HIDDEN_DIM * HIDDEN_DIM) +      /* W_hr */ \
    (HIDDEN_DIM) +                   /* b_ir */ \
    (HIDDEN_DIM) +                   /* b_hr */ \
    (HIDDEN_DIM * EMBED_DIM) +       /* W_iz */ \
    (HIDDEN_DIM * HIDDEN_DIM) +      /* W_hz */ \
    (HIDDEN_DIM) +                   /* b_iz */ \
    (HIDDEN_DIM) +                   /* b_hz */ \
    (HIDDEN_DIM * EMBED_DIM) +       /* W_in */ \
    (HIDDEN_DIM * HIDDEN_DIM) +      /* W_hn */ \
    (HIDDEN_DIM) +                   /* b_in */ \
    (HIDDEN_DIM) +                   /* b_hn */ \
    (VOCAB_SIZE * HIDDEN_DIM) +      /* W_out */ \
    (VOCAB_SIZE)                     /* b_out */ \
)

/* 4 bytes for scale + int8 weights */
#define MODEL_TOTAL_SIZE (sizeof(float) + MODEL_NUM_WEIGHTS)

bool model_load(uint8_t* buffer);
bool model_exists(void);
const uint8_t* model_get_archive_ptr(void);

#endif /* MODEL_IO_H */
