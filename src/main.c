#include <debug.h>
#include <graphx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ti/getcsc.h>
#include <ti/screen.h>

#include "gru.h"
#include "memory.h"
#include "model_io.h"

#define MAX_GEN_CHARS 120
#define ENABLE_DIAGNOSTICS 1
/* Screen dimensions for text in 8bpp mode (approx 8x8 font) */
#define GFX_SCREEN_COLS (320 / 8)
#define GFX_SCREEN_ROWS (240 / 10)
#define FONT_HEIGHT 10

static const char *prompts[] = {"The ", "To be ", "I think ", "Hello ",
                                "Once "};
#define NUM_PROMPTS 5

static void print_text_at(int x, int y, const char *str) {
  gfx_SetTextXY(x, y);
  gfx_PrintString(str);
}

static void print_centered(int y, const char *str) {
  int w = gfx_GetStringWidth(str);
  gfx_SetTextXY((320 - w) / 2, y);
  gfx_PrintString(str);
}

static GRU_Model model;
static uint8_t *weights_buffer = NULL;
static int32_t output_logits[VOCAB_SIZE];

/* Simple text cursor for continuous printing */
static int cursor_x = 0;
static int cursor_y = 0;

static void newline(void) {
  cursor_x = 0;
  cursor_y += FONT_HEIGHT;
  if (cursor_y > 230) {
    cursor_y = 0;
    gfx_FillScreen(255); /* Clear with white */
    gfx_SetTextXY(0, 0);
  }
}

static void print_char_gfx(char c) {
  char str[2] = {c, '\0'};

  if (c == '\n') {
    newline();
    return;
  }

  if (cursor_x + 8 > 320) {
    newline();
  }

  gfx_SetTextXY(cursor_x, cursor_y);
  gfx_PrintString(str);
  cursor_x += gfx_GetStringWidth(str);
}

static void print_string_gfx(const char *str) {
  while (*str) {
    print_char_gfx(*str);
    str++;
  }
}

#if ENABLE_DIAGNOSTICS
static void run_diagnostics(GRU_Model *m) {
  char buf[64];
  int32_t output[VOCAB_SIZE];

  gfx_FillScreen(255);
  cursor_x = 0;
  cursor_y = 0;

  print_string_gfx("=== DIAGNOSTICS ===");
  newline();

  /* Show weight loading verification */
  /* embed is const int8_t*, cast to int for printing */
  /* Note: embed might be NULL if load failed, but we checked that */
  sprintf(buf, "embed size: %d", (int)MODEL_TOTAL_SIZE);
  print_string_gfx(buf);
  newline();

  if (m->embed) {
    sprintf(buf, "embed[0][0..1]=%d,%d", (int)m->embed[0], (int)m->embed[1]);
    print_string_gfx(buf);
    newline();
  } else {
    print_string_gfx("embed is NULL!");
    newline();
  }

  print_string_gfx("Checking b_out...");
  newline();
  if (m->b_out) {
    /* Print address for debugging */
    sprintf(buf, "b_out ptr: %p", (void *)m->b_out);
    print_string_gfx(buf);
    newline();

    sprintf(buf, "b_out[0]=%d", (int)m->b_out[0]);
    print_string_gfx(buf);
    newline();
  } else {
    print_string_gfx("b_out is NULL!");
    newline();
  }

  print_string_gfx("Running Forward...");
  newline();

  /* Process "The " and show hidden state */
  gru_reset_hidden(m);
  gru_forward(m, 52, output); /* 'T' */
  gru_forward(m, 72, output); /* 'h' */
  gru_forward(m, 69, output); /* 'e' */
  gru_forward(m, 0, output);  /* ' ' */

  sprintf(buf, "h[0..2]=%d,%d,%d", (int)m->hidden[0], (int)m->hidden[1],
          (int)m->hidden[2]);
  print_string_gfx(buf);
  newline();

  sprintf(buf, "out[0]' '=%ld", (long)output[0]);
  print_string_gfx(buf);
  newline();

  sprintf(buf, "out[69]'e'=%ld", (long)output[69]);
  print_string_gfx(buf);
  newline();

  print_string_gfx("[Press Key]");
  while (!os_GetCSC())
    ;
}
#endif

static void generate_text(const char *seed, uint8_t max_chars) {
  uint8_t i;
  uint8_t next_idx;
  char c;
  uint8_t chars_generated = 0;
  uint16_t rand_seed = 12345; /* Random seed */

  gfx_FillScreen(255);
  cursor_x = 0;
  cursor_y = 0;

  print_string_gfx("Seed: ");
  print_string_gfx(seed);
  newline();
  newline();
  print_string_gfx("Output:");
  newline();

  /* 1. Feed the seed prompt */
  print_string_gfx(seed);
  gru_reset_hidden(&model);

  /* Process seed characters */
  for (i = 0; seed[i] != '\0'; i++) {
    gru_forward(&model, (uint8_t)seed[i], output_logits);
  }

  /* Get last char to continue randomness */
  rand_seed += (uint8_t)seed[i - 1];

  /* 2. Generate new text */
  while (chars_generated < max_chars) {
    /* Sample next character using top-k=4 */
    next_idx = gru_sample_topk(output_logits, 4, rand_seed++);
    c = (char)next_idx;

    print_char_gfx(c);

    /* Stop at newline if desired, or just continue */
    /* if (c == '\n') break; */

    /* Feed back into model */
    gru_forward(&model, next_idx, output_logits);
    chars_generated++;
  }

  newline();
  newline();
  print_string_gfx("[Press any key]");
  while (!os_GetCSC())
    ;
}

static int8_t show_menu(void) {
  int key;

  gfx_FillScreen(255);

  gfx_SetTextFGColor(0); /* Black text */
  print_centered(10, "Calculator LLM v0.6");
  print_centered(25, "128-hidden (120KB)");

  print_text_at(10, 50, "Select Prompt:");

  print_text_at(20, 70, "1. The ...");
  print_text_at(20, 85, "2. To be ...");
  print_text_at(20, 100, "3. I think ...");
  print_text_at(20, 115, "4. Hello ...");
  print_text_at(20, 130, "5. Once ...");
  print_text_at(20, 145, "6. Diagnostics");

  print_centered(165, "Press 1-6 to generate");
  print_centered(180, "Press Clear to exit");

  while ((key = os_GetCSC()) == 0)
    ;

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
  if (key == sk_6)
    return 5; /* 5 is mapped to Diagnostics */

  return -1;
}

int main(void) {
  const uint8_t *archive_ptr;
  int8_t selection;
  char buf[64];

  /* Initialize system */
  mem_init(); /* Enters 8bpp graphics mode */

  /* Setup clean white background and black text */
  /* gfx_SetPalette(palette_ptr, size, offset) - passing gfx_palette (default)
   */
  gfx_SetPalette(gfx_palette, 256, 0);
  /* Actually mem_init does gfx_SetDrawBuffer, so let's check palette.
     Default palette has 255 as white, 0 as black normally,
     but let's just assume standard palette or use palette 0 (Black) and 255
     (White) safely? Standard 8bpp palette: 255 is usually white. */
  gfx_FillScreen(255);
  gfx_SetTextFGColor(0);
  gfx_SetTextBGColor(255);
  // gfx_SetTextTransparentColor(255); // Optional

  print_centered(100, "Loading Calculator LLM...");

  /* Load Model Logic */
  if ((archive_ptr = model_get_archive_ptr()) != NULL) {
    gru_init(&model, archive_ptr);
  } else if (model_exists_split()) {
    print_centered(120, "Loading Split Model (120KB)...");

    sprintf(buf, "Mn: &model=%p", (void *)&model);
    gfx_SetTextXY(0, 10);
    gfx_PrintString(buf);

    /* Distributed load: Allocates tensors individually via model_io.c */
    if (!model_load_distributed(&model)) {
      print_centered(140, "Error: RAM/Load Failed");
      mem_print_stats(); /* Only visible if debug enabled but good practice */
      while (!os_GetCSC())
        ;
      goto exit;
    }

    gfx_SetTextXY(0, 30);
    sprintf(buf, "Mn: b=%p &b=%p", (void *)model.b_out, (void *)&model.b_out);
    print_string_gfx(buf);
    newline();
    while (!os_GetCSC())
      ; /* Pause to see it */

    /* Initialize model struct (reset hidden state, pointers already set) */
    gru_init(&model, NULL);
  } else if (model_exists_split()) {
    print_centered(120, "Loading Split Model (120KB)...");

    sprintf(buf, "Mn: &model=%p", (void *)&model);
    gfx_SetTextXY(0, 10);
    gfx_PrintString(buf);

    /* Distributed load: Allocates tensors individually via model_io.c */
    if (!model_load_distributed(&model)) {
      print_centered(140, "Error: RAM/Load Failed");
      mem_print_stats();
      while (!os_GetCSC())
        ;
      goto exit;
    }

    gfx_SetTextXY(0, 30);
    sprintf(buf, "Mn: b=%p &b=%p", (void *)model.b_out, (void *)&model.b_out);
    print_string_gfx(buf);
    newline();
    while (!os_GetCSC())
      ; /* Pause to see it */

    /* Initialize model struct (reset hidden state, pointers already set) */
    gru_init(&model, NULL);
  } else if (model_exists()) {
    /* Fallback for single AppVar in RAM */
    weights_buffer = (uint8_t *)tensor_alloc(MODEL_TOTAL_SIZE);
    if (!weights_buffer || !model_load(weights_buffer)) {
      print_centered(140, "Error: Load Failed");
      while (!os_GetCSC())
        ;
      goto exit;
    }
    gru_init(&model, weights_buffer);
  } else {
    print_centered(140, "Error: No Model Found");
    print_centered(155, "Transfer GRUMDL1/2");
    while (!os_GetCSC())
      ;
    goto exit;
  }

  /* Main Loop */
  while (1) {
    selection = show_menu();
    if (selection < 0)
      break;

    if (selection == 5) {
#if ENABLE_DIAGNOSTICS
      run_diagnostics(&model);
#endif
    } else {
      generate_text(prompts[selection], MAX_GEN_CHARS);
    }
  }

exit:
  mem_cleanup(); /* Exits graphics mode */
  return 0;
}
