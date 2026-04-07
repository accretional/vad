#!/bin/bash
set -euo pipefail

# Full pipeline: encode audio to 16k f32, run VAD, save outputs, re-encode to MP3.
#
# Usage: ./cmd/pkg-example/run.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="$SCRIPT_DIR/input"
SEG_DIR="$SCRIPT_DIR/segmented-output"
UNSEG_DIR="$SCRIPT_DIR/unsegmented-output"

# Clean previous output
rm -rf "$INPUT_DIR" "$SEG_DIR" "$UNSEG_DIR"
mkdir -p "$INPUT_DIR" "$SEG_DIR" "$UNSEG_DIR"

# Step 1: Encode source audio to 16kHz mono f32
echo "=== Encoding audio to 16kHz f32 ==="
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg is not installed."
    exit 1
fi

for mp3 in data/*.mp3; do
    [ -f "$mp3" ] || continue
    base=$(basename "${mp3%.mp3}")
    out="$INPUT_DIR/${base}-16k.f32"
    echo "  $mp3 -> $out"
    ffmpeg -hide_banner -loglevel error -i "$mp3" \
        -ar 16000 -ac 1 -f f32le -acodec pcm_f32le "$out"
done

# Step 2: Build
echo ""
echo "=== Building pkg-example ==="
go build -o "$SCRIPT_DIR/pkg-example" ./cmd/pkg-example/

# Step 3: Detect ORT library
if [ -z "${ONNXRUNTIME_LIB:-}" ]; then
    ARCH=$(uname -m)
    OS=$(uname -s)
    if [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            PLATFORM="osx-arm64"
        else
            PLATFORM="osx-x86_64"
        fi
        DYLD_LIBRARY_PATH="third_party/onnxruntime-${PLATFORM}-1.22.0/lib:${DYLD_LIBRARY_PATH:-}"
        export DYLD_LIBRARY_PATH
        export ONNXRUNTIME_LIB="third_party/onnxruntime-${PLATFORM}-1.22.0/lib/libonnxruntime.dylib"
    else
        PLATFORM="linux-x64"
        LD_LIBRARY_PATH="third_party/onnxruntime-${PLATFORM}-1.22.0/lib:${LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH
        export ONNXRUNTIME_LIB="third_party/onnxruntime-${PLATFORM}-1.22.0/lib/libonnxruntime.so"
    fi
fi

# Step 4: Run inference + segmentation
echo ""
echo "=== Running VAD on all audio files ==="
"$SCRIPT_DIR/pkg-example" -data "$INPUT_DIR"

# Step 5: Re-encode as MP3
echo ""
echo "=== Re-encoding segmented output to MP3 ==="
./encode-to-16k.sh --reverse "$SEG_DIR"

echo ""
echo "=== Re-encoding unsegmented output to MP3 ==="
./encode-to-16k.sh --reverse "$UNSEG_DIR"

echo ""
echo "=== Complete ==="
echo "Input (16k f32): $INPUT_DIR/"
echo "Segmented output: $SEG_DIR/"
echo "Unsegmented output: $UNSEG_DIR/"
echo ""
echo "Segmented MP3s:"
ls -lh "$SEG_DIR"/*.mp3 2>/dev/null || echo "  (none)"
echo ""
echo "Unsegmented MP3s:"
ls -lh "$UNSEG_DIR"/*.mp3 2>/dev/null || echo "  (none)"
