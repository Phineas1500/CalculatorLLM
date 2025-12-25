#include "model_io.h"
#include <debug.h>
#include <fileioc.h>

bool model_load(uint8_t *buffer) {
  uint8_t handle;
  size_t bytes_read;

  dbg_printf("model_load: opening '%s'\n", MODEL_APPVAR_NAME);
  handle = ti_Open(MODEL_APPVAR_NAME, "r");
  if (handle == 0) {
    dbg_printf("model_load: ti_Open failed\n");
    return false;
  }
  dbg_printf("model_load: handle=%d\n", handle);

  ti_Seek(2, SEEK_SET, handle);

  bytes_read = ti_Read(buffer, 1, MODEL_TOTAL_SIZE, handle);
  dbg_printf("model_load: read %lu of %lu bytes\n", (unsigned long)bytes_read,
             (unsigned long)MODEL_TOTAL_SIZE);

  ti_Close(handle);

  return (bytes_read == MODEL_TOTAL_SIZE);
}

bool model_exists(void) {
  uint8_t handle;

  dbg_printf("model_exists: checking '%s'\n", MODEL_APPVAR_NAME);
  handle = ti_Open(MODEL_APPVAR_NAME, "r");
  if (handle == 0) {
    dbg_printf("model_exists: not found\n");
    return false;
  }

  dbg_printf("model_exists: found (handle=%d)\n", handle);
  ti_Close(handle);
  return true;
}

const uint8_t *model_get_archive_ptr(void) {
  uint8_t handle;
  uint8_t *ptr;

  dbg_printf("model_get_archive_ptr: opening '%s'\n", MODEL_APPVAR_NAME);
  handle = ti_Open(MODEL_APPVAR_NAME, "r");
  if (handle == 0) {
    dbg_printf("model_get_archive_ptr: ti_Open failed\n");
    return NULL;
  }

  dbg_printf("model_get_archive_ptr: checking if archived\n");
  if (!ti_IsArchived(handle)) {
    dbg_printf("model_get_archive_ptr: not archived, using RAM path\n");
    ti_Close(handle);
    return NULL;
  }

  ptr = (uint8_t *)ti_GetDataPtr(handle);
  /* ti_GetDataPtr returns pointer to AppVar data content - no extra skip needed
   */

  dbg_printf("model_get_archive_ptr: archive ptr=%p\n", (void *)ptr);
  ti_Close(handle);

  return ptr;
}
