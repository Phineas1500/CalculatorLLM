#include "model_io.h"
#include "gru.h"
#include "memory.h"
#include <debug.h>
#include <fileioc.h>
#include <graphx.h>
#include <stdio.h>
#include <string.h>

/* Helper to read data from split AppVars seamlessly */
typedef struct {
  uint8_t handle1;
  uint8_t handle2;
  uint32_t offset; /* Current global offset */
} SplitReader;

static bool reader_init(SplitReader *r) {
  r->offset = 0;

  r->handle1 = ti_Open(MODEL_APPVAR_PART1, "r");
  if (r->handle1 == 0)
    return false;

  r->handle2 = ti_Open(MODEL_APPVAR_PART2, "r");
  if (r->handle2 == 0) {
    ti_Close(r->handle1);
    return false;
  }

  /* No seek needed - ti_Open points to data start */
  return true;
}

static void reader_close(SplitReader *r) {
  if (r->handle1)
    ti_Close(r->handle1);
  if (r->handle2)
    ti_Close(r->handle2);
}

static bool reader_read(SplitReader *r, void *dest, size_t size) {
  size_t bytes_left = size;
  uint8_t *ptr = (uint8_t *)dest;
  /* Read in smaller chunks for safety */
  size_t max_chunk = 4096;

  while (bytes_left > 0) {
    size_t chunk = (bytes_left > max_chunk) ? max_chunk : bytes_left;
    uint8_t handle;

    if (r->offset < MODEL_SPLIT_SIZE) {
      /* Read from Part 1 */
      size_t avail = MODEL_SPLIT_SIZE - r->offset;
      if (chunk > avail)
        chunk = avail;
      handle = r->handle1;
    } else {
      /* Read from Part 2 */
      handle = r->handle2;
    }

    /* Ensure we don't try to read 0 bytes if hitting boundary exactly */
    if (chunk == 0 && r->offset == MODEL_SPLIT_SIZE) {
      continue; /* Loop again, will fall into Part 2 */
    }

    if (ti_Read(ptr, 1, chunk, handle) != chunk) {
      return false;
    }

    ptr += chunk;
    bytes_left -= chunk;
    r->offset += chunk;
  }
  return true;
}

static bool load_tensor(SplitReader *r, int8_t **ptr_ref, size_t size) {
  /* Allocate memory for this tensor */
  *ptr_ref = (int8_t *)tensor_alloc(size);
  if (*ptr_ref == NULL) {
    /* OOM */
    char buf[32];
    gfx_SetTextXY(0, 50);
    sprintf(buf, "Alloc Fail: %u", size);
    gfx_PrintString(buf);
    return false;
  }

  /* Read data into it */
  if (!reader_read(r, *ptr_ref, size)) {
    return false;
  }

  return true;
}

bool model_load_distributed(GRU_Model *m) {
  SplitReader reader;
  char buf[64];

  if (!reader_init(&reader)) {
    return false;
  }

  /* Read global scale (4 bytes) - stored but not used in int math */
  if (!reader_read(&reader, &m->scale, 4)) {
    reader_close(&reader);
    return false;
  }

  /* Load all tensors in order matching train.py export */
  /* embed */
  if (!load_tensor(&reader, &m->embed, VOCAB_SIZE * EMBED_DIM))
    goto error;

  /* GRU Layer 1 - Reset gate */
  if (!load_tensor(&reader, &m->W_ir, HIDDEN_DIM * EMBED_DIM))
    goto error;
  if (!load_tensor(&reader, &m->W_hr, HIDDEN_DIM * HIDDEN_DIM))
    goto error;
  if (!load_tensor(&reader, &m->b_ir, HIDDEN_DIM))
    goto error;
  if (!load_tensor(&reader, &m->b_hr, HIDDEN_DIM))
    goto error;

  /* GRU Layer 1 - Update gate */
  if (!load_tensor(&reader, &m->W_iz, HIDDEN_DIM * EMBED_DIM))
    goto error;
  if (!load_tensor(&reader, &m->W_hz, HIDDEN_DIM * HIDDEN_DIM))
    goto error;
  if (!load_tensor(&reader, &m->b_iz, HIDDEN_DIM))
    goto error;
  if (!load_tensor(&reader, &m->b_hz, HIDDEN_DIM))
    goto error;

  /* GRU Layer 1 - New gate */
  if (!load_tensor(&reader, &m->W_in, HIDDEN_DIM * EMBED_DIM))
    goto error;
  if (!load_tensor(&reader, &m->W_hn, HIDDEN_DIM * HIDDEN_DIM))
    goto error;
  if (!load_tensor(&reader, &m->b_in, HIDDEN_DIM))
    goto error;
  if (!load_tensor(&reader, &m->b_hn, HIDDEN_DIM))
    goto error;

  /* Output Layer */
  if (!load_tensor(&reader, &m->W_out, VOCAB_SIZE * HIDDEN_DIM))
    goto error;
  if (!load_tensor(&reader, &m->b_out, VOCAB_SIZE))
    goto error;

  /* Visual Debugging - Success */
  gfx_SetTextXY(0, 20);
  sprintf(buf, "IO: m=%p b=%p", (void *)m, (void *)m->b_out);
  gfx_PrintString(buf);

  /* Debounce and Pause */
  while (os_GetCSC())
    ; /* Wait for key release */
  while (!os_GetCSC())
    ; /* Wait for key press */

  reader_close(&reader);
  /* dbg_printf("model_load_distributed: success\n"); */
  return true;

error:
  reader_close(&reader);
  gfx_SetTextXY(0, 100);
  gfx_PrintString("IO: Load Error!");
  while (os_GetCSC())
    ;
  while (!os_GetCSC())
    ;
  return false;
}

/* Include existing functions for backward compatibility/reference */
bool model_exists(void) {
  uint8_t handle = ti_Open(MODEL_APPVAR_NAME, "r");
  if (handle) {
    ti_Close(handle);
    return true;
  }
  return false;
}

bool model_exists_split(void) {
  uint8_t h1 = ti_Open(MODEL_APPVAR_PART1, "r");
  if (!h1)
    return false;
  ti_Close(h1);

  uint8_t h2 = ti_Open(MODEL_APPVAR_PART2, "r");
  if (!h2)
    return false;
  ti_Close(h2);

  return true;
}

/* Legacy single-appvar loader (unused for 128-hidden but kept for logical
 * completeness) */
bool model_load(uint8_t *buffer) {
  return false; /* Not used in distributed mode */
}
const uint8_t *model_get_archive_ptr(void) { return NULL; }
