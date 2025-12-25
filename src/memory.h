/*
 * memory.h - Custom multi-heap memory allocator for TI-84 Plus CE
 *
 * Uses three memory regions to maximize available RAM:
 * 1. Default heap (~10-20KB)
 * 2. User RAM (~100KB) - from os_GetFreeRAM()
 * 3. VRAM (75KB) - freed by using 8bpp graphics mode
 *
 * Based on techniques from: https://z80.me/blog/calculator-ai-part-1/
 */

#ifndef MEMORY_H
#define MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Memory pool structure for tracking allocations */
typedef struct {
  uint8_t *base; /* Base address of the pool */
  size_t size;   /* Total size of the pool */
  size_t used;   /* Amount currently allocated */
} MemPool;

/* Initialize all memory pools - call this at startup */
void mem_init(void);

/* Clean up memory pools - call before exit */
void mem_cleanup(void);

/* Get total available memory across all pools */
size_t mem_available(void);

/* Get total memory in use */
size_t mem_used(void);

/* Allocate from the tensor heap (multi-pool) */
void *tensor_alloc(size_t size);

/* Free tensor memory (note: simple bump allocator, free is no-op for now) */
void tensor_free(void *ptr);

/* Reset all tensor allocations (call between inferences if needed) */
void tensor_reset(void);

/* Debug: print memory stats */
void mem_print_stats(void);

#endif /* MEMORY_H */
