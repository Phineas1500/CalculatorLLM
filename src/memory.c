/*
 * memory.c - Custom multi-heap memory allocator for TI-84 Plus CE
 *
 * Implements a simple bump allocator across multiple memory regions
 * to maximize available RAM for neural network weights and activations.
 *
 * Based on techniques from: https://z80.me/blog/calculator-ai-part-1/
 */

#include "memory.h"
#include <debug.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ez80__
/* TI-84 Plus CE specific includes */
#include <graphx.h>
#include <sys/lcd.h>

/*
 * Memory Map for TI-84 Plus CE:
 *
 * lcd_Ram (0xD40000): Start of VRAM
 *   - 8bpp mode uses 320*240 = 76,800 bytes (0xD40000 - 0xD52C00)
 *   - The rest (76,800 bytes) can be used as heap!
 *
 * We use gfx_SetDrawBuffer() to use single buffering, which means
 * the "back buffer" memory becomes available for our use.
 */

#define VRAM_BASE ((uint8_t *)lcd_Ram)
#define VRAM_8BPP_SIZE (320 * 240) /* 76,800 bytes for screen */
#define VRAM_HEAP_BASE (VRAM_BASE + VRAM_8BPP_SIZE)
#define VRAM_HEAP_SIZE (75 * 1024) /* ~75KB for our heap */

/*
 * User RAM: Start conservatively with 60KB static buffer.
 * This gets placed in the BSS section which uses available RAM.
 * Can increase after testing shows it works.
 */
#define STATIC_POOL_SIZE (58 * 1024) /* 58KB static buffer */
static uint8_t static_pool[STATIC_POOL_SIZE];

#else
/* PC fallback - just use static buffers */
#define STATIC_POOL_SIZE (256 * 1024)
#define VRAM_HEAP_SIZE (75 * 1024)
static uint8_t static_pool[STATIC_POOL_SIZE];
static uint8_t vram_pool_buffer[VRAM_HEAP_SIZE];
#endif

/* Memory pools */
static MemPool static_mem_pool;
static MemPool vram_pool;
static bool initialized = false;
static bool gfx_active = false;

void mem_init(void) {
  if (initialized)
    return;

  dbg_printf("mem_init: Initializing memory pools\n");

  /* Static pool - always available */
  static_mem_pool.base = static_pool;
  static_mem_pool.size = STATIC_POOL_SIZE;
  static_mem_pool.used = 0;

#ifdef __ez80__
  /* Initialize graphics in 8bpp mode to free up VRAM */
  gfx_Begin();
  /* gfx_SetDrawBuffer(); REMOVED - we want to draw to screen, and use back
   * buffer as heap */
  gfx_active = true;

  /* VRAM heap - the freed "back buffer" area */
  vram_pool.base = VRAM_HEAP_BASE;
  vram_pool.size = VRAM_HEAP_SIZE;
  vram_pool.used = 0;

  dbg_printf("  Static pool: base=%p, size=%lu\n", (void *)static_mem_pool.base,
             (unsigned long)static_mem_pool.size);
  dbg_printf("  VRAM heap: base=%p, size=%lu\n", (void *)vram_pool.base,
             (unsigned long)vram_pool.size);
#else
  /* PC fallback */
  vram_pool.base = vram_pool_buffer;
  vram_pool.size = VRAM_HEAP_SIZE;
  vram_pool.used = 0;
#endif

  initialized = true;
  mem_print_stats();
}

void mem_cleanup(void) {
  if (!initialized)
    return;

  dbg_printf("mem_cleanup: Releasing memory pools\n");

#ifdef __ez80__
  if (gfx_active) {
    gfx_End();
    gfx_active = false;
  }
#endif

  static_mem_pool.used = 0;
  vram_pool.used = 0;

  initialized = false;
}

size_t mem_available(void) {
  if (!initialized)
    return 0;

  size_t static_avail = static_mem_pool.size - static_mem_pool.used;
  size_t vram_avail = vram_pool.size - vram_pool.used;

  return static_avail + vram_avail;
}

size_t mem_used(void) {
  if (!initialized)
    return 0;
  return static_mem_pool.used + vram_pool.used;
}

void *tensor_alloc(size_t size) {
  if (!initialized) {
    dbg_printf("tensor_alloc: ERROR - not initialized!\n");
    return NULL;
  }

  /* Align to 4-byte boundary for ez80 */
  size = (size + 3) & ~3;

  /* Try static pool first */
  if (static_mem_pool.used + size <= static_mem_pool.size) {
    void *ptr = static_mem_pool.base + static_mem_pool.used;
    static_mem_pool.used += size;
    return ptr;
  }

  /* Fall back to VRAM heap */
  if (vram_pool.used + size <= vram_pool.size) {
    void *ptr = vram_pool.base + vram_pool.used;
    vram_pool.used += size;
    return ptr;
  }

  /* Last resort: try standard malloc */
  {
    void *ptr = malloc(size);
    if (ptr) {
      /* Record it so we can free it later if needed (simple hack) */
      /* In a production allocator we'd list track, but here we likely won't
       * reset often */
      dbg_printf("tensor_alloc: using malloc for %u bytes\n", size);
      return ptr;
    }
  }

  dbg_printf("tensor_alloc: FAILED - out of memory for %lu bytes\n",
             (unsigned long)size);
  return NULL;
}

void tensor_free(void *ptr) {
  /* Simple bump allocator - free is a no-op */
  /* Use tensor_reset() to free all at once */
  (void)ptr;
}

void tensor_reset(void) {
  if (!initialized)
    return;

  static_mem_pool.used = 0;
  vram_pool.used = 0;
}

void mem_print_stats(void) {
  size_t total_size = static_mem_pool.size + vram_pool.size;
  size_t total_used = static_mem_pool.used + vram_pool.used;

  dbg_printf("=== Memory Stats ===\n");
  dbg_printf("Static: %luK / %luK\n",
             (unsigned long)(static_mem_pool.used / 1024),
             (unsigned long)(static_mem_pool.size / 1024));
  dbg_printf("VRAM:   %luK / %luK\n", (unsigned long)(vram_pool.used / 1024),
             (unsigned long)(vram_pool.size / 1024));
  dbg_printf("Total:  %luK / %luK available\n",
             (unsigned long)((total_size - total_used) / 1024),
             (unsigned long)(total_size / 1024));
}
