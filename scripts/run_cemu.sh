#!/bin/bash
# Run CALCLLM in CEmu emulator
# Usage: ./scripts/run_cemu.sh [--autotest]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CEMU="/Applications/CEmu.app/Contents/MacOS/CEmu"
ROM="$PROJECT_DIR/CalcData/image.rom"
PROGRAM="$PROJECT_DIR/bin/CALCLLM.8xp"
MODEL="$PROJECT_DIR/training/GRUMODEL.8xv"
CLIBS="$PROJECT_DIR/CalcData/clibs.8xg"

# Verify CEmu exists
if [ ! -f "$CEMU" ]; then
    echo "ERROR: CEmu not found at $CEMU"
    exit 1
fi

# Verify required files
for file in "$ROM" "$PROGRAM" "$MODEL" "$CLIBS"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    fi
done

echo "=== CEmu Calculator LLM Runner ==="
echo "ROM:     $ROM"
echo "Program: $PROGRAM"
echo "Model:   $MODEL"
echo "C Libs:  $CLIBS"
echo ""

if [ "$1" == "--autotest" ]; then
    echo "Running in autotest mode..."
    "$CEMU" \
        -t "$PROJECT_DIR/autotest.json" \
        --no-test-dialog
else
    echo "Launching CEmu with files..."
    echo "(Files will be sent to RAM, program will auto-launch)"
    echo ""
    "$CEMU" \
        -r "$ROM" \
        -m "$PROGRAM" \
        -s "$MODEL" \
        -s "$CLIBS" \
        --launch CALCLLM
fi
