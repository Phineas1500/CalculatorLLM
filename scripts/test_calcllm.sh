#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CEMU="/Applications/CEmu.app/Contents/MacOS/CEmu"
ROM="$PROJECT_DIR/CalcData/image.rom"
PROGRAM="$PROJECT_DIR/bin/CALCLLM.8xp"
MODEL="$PROJECT_DIR/training/GRUMODEL.8xv"
CLIBS="$PROJECT_DIR/CalcData/clibs.8xg"
SCREENSHOT_DIR="$PROJECT_DIR/screenshots"

mkdir -p "$SCREENSHOT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCREENSHOT="$SCREENSHOT_DIR/test_$TIMESTAMP.png"

echo "=== CALCLLM Test Runner ==="
echo "Timestamp: $TIMESTAMP"
echo ""

for file in "$ROM" "$PROGRAM" "$MODEL" "$CLIBS"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    fi
done

echo "Launching CEmu in background..."
"$CEMU" \
    -r "$ROM" \
    -m "$PROGRAM" \
    -s "$MODEL" \
    -s "$CLIBS" \
    --launch CALCLLM \
    -c "test_session" &
CEMU_PID=$!

echo "CEmu PID: $CEMU_PID"
echo "Waiting for program to start (5s)..."
sleep 5

echo "Sending key '1' to select first prompt..."
"$CEMU" -c "test_session" -m /dev/null 2>/dev/null || true
sleep 1

echo "Waiting for text generation (20s)..."
sleep 20

echo "Taking screenshot..."
"$CEMU" -c "test_session" --screenshot "$SCREENSHOT" 2>/dev/null || true
sleep 1

echo "Stopping CEmu..."
kill $CEMU_PID 2>/dev/null || true

if [ -f "$SCREENSHOT" ]; then
    echo ""
    echo "Screenshot saved: $SCREENSHOT"
    echo "Open with: open \"$SCREENSHOT\""
else
    echo "WARNING: Screenshot not captured"
fi

echo ""
echo "Test complete."
