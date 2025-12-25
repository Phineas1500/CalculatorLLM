# ----------------------------
# Makefile Options
# ----------------------------

NAME = CALCLLM
DESCRIPTION = "Calculator LLM"
COMPRESSED = NO

CFLAGS = -Wall -Wextra -Oz
CXXFLAGS = -Wall -Wextra -Oz

# ----------------------------

include $(shell cedev-config --makefile)
