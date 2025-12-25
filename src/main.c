#include <debug.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ti/getcsc.h>
#include <ti/screen.h>

#include "gru.h"
#include "model_io.h"

#define MAX_GEN_CHARS 100
#define ENABLE_DIAGNOSTICS 1
#define SCREEN_COLS 26
#define SCREEN_ROWS 10

static const char *prompts[] = {"The ", "To be or ", "Once upon ", "Hello ",
                                "I think "};
#define NUM_PROMPTS 5

#if ENABLE_DIAGNOSTICS
static void run_diagnostics(GRU_Model *m) {
  char buf[40];
  int32_t output[VOCAB_SIZE];

  os_ClrHome();
  os_PutStrLine("=== DIAGNOSTICS ===");
  os_NewLine();

  /* Show weight loading verification */
  sprintf(buf, "embed[0][0..1]=%d,%d", (int)m->embed[0], (int)m->embed[1]);
  os_PutStrLine(buf);
  sprintf(buf, "b_out[0]=%d b_out[69]=%d", (int)m->b_out[0], (int)m->b_out[69]);
  os_PutStrLine(buf);

  /* Process "The " and show hidden state */
  gru_reset_hidden(m);
  gru_forward(m, 52, output); /* 'T' */
  gru_forward(m, 72, output); /* 'h' */
  gru_forward(m, 69, output); /* 'e' */
  gru_forward(m, 0, output);  /* ' ' */

  sprintf(buf, "h[0..2]=%d,%d,%d", (int)m->hidden[0], (int)m->hidden[1],
          (int)m->hidden[2]);
  os_PutStrLine(buf);
  sprintf(buf, "h[3..5]=%d,%d,%d", (int)m->hidden[3], (int)m->hidden[4],
          (int)m->hidden[5]);
  os_PutStrLine(buf);

  /* Show key output logits */
  sprintf(buf, "out[0]' '=%ld", (long)output[0]);
  os_PutStrLine(buf);
  sprintf(buf, "out[69]'e'=%ld", (long)output[69]);
  os_PutStrLine(buf);

  os_NewLine();
  os_PutStrLine("Press any key...");
  while (!os_GetCSC())
    ;
}
#endif

static GRU_Model model;
static uint8_t *weights_buffer = NULL;
static int32_t output_logits[VOCAB_SIZE];

static uint8_t cursor_row = 0;
static uint8_t cursor_col = 0;

static void print_char_wrapped(char c) {
  char str[2] = {c, '\0'};

  if (c == '\n' || cursor_col >= SCREEN_COLS) {
    cursor_col = 0;
    cursor_row++;
    if (cursor_row >= SCREEN_ROWS) {
      cursor_row = 0;
      os_ClrHome();
    }
    if (c == '\n')
      return;
  }

  os_SetCursorPos(cursor_row, cursor_col);
  os_PutStrFull(str);
  cursor_col++;
}

static void print_string(const char *str) {
  while (*str) {
    print_char_wrapped(*str);
    str++;
  }
}

static void generate_text(const char *seed, uint8_t max_chars) {
  uint8_t i;
  uint8_t next_idx;
  char c;
  uint8_t chars_generated = 0;
  uint16_t rand_seed = 12345; /* Random seed, increments each char */
  sk_key_t key;

  dbg_printf("generate_text: seed='%s', max=%d\n", seed, max_chars);

  gru_reset_hidden(&model);

  print_string("Seed: ");
  print_string(seed);
  print_string("\n\nOutput:\n");

  dbg_printf("Feeding seed through model...\n");
  for (i = 0; seed[i] != '\0'; i++) {
    uint8_t idx = char_to_idx(seed[i]);
    gru_forward(&model, idx, output_logits);
    print_char_wrapped(seed[i]);
  }
  dbg_printf("Seed processed, starting generation\n");

  while (chars_generated < max_chars) {
    key = os_GetCSC();
    if (key == sk_Clear || key == sk_Enter) {
      dbg_printf("User aborted generation\n");
      break;
    }

    next_idx = gru_sample_topk(output_logits, 5, rand_seed);
    rand_seed += 7919; /* Prime increment for variety */
    c = idx_to_char(next_idx);
    print_char_wrapped(c);

#if ENABLE_DIAGNOSTICS
    if (chars_generated < 3) {
      int32_t max_l = output_logits[0];
      int max_i = 0;
      int j;
      for (j = 1; j < VOCAB_SIZE; j++) {
        if (output_logits[j] > max_l) {
          max_l = output_logits[j];
          max_i = j;
        }
      }
      dbg_printf("Gen[%d]: idx=%d max_logit=%ld\n", chars_generated, max_i,
                 (long)max_l);
    }
#endif

    if (chars_generated < 5) {
      dbg_printf("Generated[%d]: idx=%d char='%c'\n", chars_generated, next_idx,
                 c);
    }

    gru_forward(&model, next_idx, output_logits);

    chars_generated++;
  }
  dbg_printf("Generated %d characters total\n", chars_generated);
}

static int8_t show_menu(void) {
  uint8_t i;
  sk_key_t key;

  os_ClrHome();
  os_PutStrLine("=== Calculator LLM ===");
  os_NewLine();
  os_PutStrLine("Select a prompt:");
  os_NewLine();

  for (i = 0; i < NUM_PROMPTS; i++) {
    char line[30];
    sprintf(line, "%d. \"%s...\"", i + 1, prompts[i]);
    os_PutStrLine(line);
  }

  os_NewLine();
  os_PutStrLine("Press 1-5 or CLEAR to exit");

  while (1) {
    key = os_GetCSC();
    if (key == sk_Clear) {
      return -1;
    }
    if (key == sk_1)
      return 0;
    if (key == sk_2)
      return 1;
    if (key == sk_3)
      return 2;
    if (key == sk_4)
      return 3;
    if (key == sk_5)
      return 4;
  }
}

int main(void) {
  const uint8_t *archive_ptr;
  int8_t selection;
  char size_str[30];

  dbg_printf("=== CALCLLM Starting ===\n");
  dbg_printf("MODEL_TOTAL_SIZE = %lu bytes\n", (unsigned long)MODEL_TOTAL_SIZE);

  os_ClrHome();
  os_PutStrLine("Calculator LLM v0.4");
  os_PutStrLine("(pure integer Q15)");
  os_NewLine();

  sprintf(size_str, "Model size: %lu KB",
          (unsigned long)(MODEL_TOTAL_SIZE / 1024));
  os_PutStrLine(size_str);
  os_PutStrLine("Loading model...");
  dbg_printf("Attempting to load model...\n");

  dbg_printf("Checking for archived model...\n");
  archive_ptr = model_get_archive_ptr();

  if (archive_ptr != NULL) {
    dbg_printf("Found archived model at %p\n", (void *)archive_ptr);
    os_PutStrLine("Using archived model");
    gru_init(&model, archive_ptr);
    dbg_printf("gru_init completed (archive)\n");
  } else {
    dbg_printf("No archived model, checking RAM...\n");
    if (!model_exists()) {
      dbg_printf("ERROR: GRUMODEL appvar not found!\n");
      os_PutStrLine("ERROR: GRUMODEL not found!");
      os_NewLine();
      os_PutStrLine("Please transfer the model");
      os_PutStrLine("AppVar to your calculator.");
      os_NewLine();
      os_PutStrLine("Press any key to exit...");
      while (!os_GetCSC())
        ;
      return 1;
    }

    dbg_printf("Model exists, allocating %lu bytes\n",
               (unsigned long)MODEL_TOTAL_SIZE);
    weights_buffer = (uint8_t *)malloc(MODEL_TOTAL_SIZE);
    if (weights_buffer == NULL) {
      dbg_printf("ERROR: malloc failed!\n");
      os_PutStrLine("ERROR: Out of memory!");
      os_NewLine();
      os_PutStrLine("Press any key to exit...");
      while (!os_GetCSC())
        ;
      return 1;
    }
    dbg_printf("Allocated buffer at %p\n", (void *)weights_buffer);

    os_PutStrLine("Loading from RAM...");
    dbg_printf("Calling model_load...\n");
    if (!model_load(weights_buffer)) {
      dbg_printf("ERROR: model_load failed!\n");
      os_PutStrLine("ERROR: Failed to load!");
      free(weights_buffer);
      while (!os_GetCSC())
        ;
      return 1;
    }
    dbg_printf("model_load succeeded\n");

    gru_init(&model, weights_buffer);
    dbg_printf("gru_init completed (RAM)\n");
  }

  dbg_printf("Model initialization complete!\n");
  os_PutStrLine("Model loaded!");
  os_NewLine();
  os_PutStrLine("Press any key...");
  while (!os_GetCSC())
    ;

#if ENABLE_DIAGNOSTICS
  run_diagnostics(&model);
#endif

  dbg_printf("Entering main loop\n");
  while (1) {
    selection = show_menu();
    if (selection < 0) {
      dbg_printf("User selected exit\n");
      break;
    }

    dbg_printf("User selected prompt %d: '%s'\n", selection,
               prompts[selection]);
    os_ClrHome();
    cursor_row = 0;
    cursor_col = 0;
    generate_text(prompts[selection], MAX_GEN_CHARS);
    dbg_printf("Generation complete\n");

    os_NewLine();
    os_NewLine();
    print_string("\n[Press any key]");
    while (!os_GetCSC())
      ;
  }

  if (weights_buffer != NULL) {
    free(weights_buffer);
  }

  return 0;
}
