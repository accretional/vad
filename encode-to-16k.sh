#!/bin/bash
set -euo pipefail

# Convert between MP3 and 16kHz mono float32 little-endian PCM (.f32).
#
# Usage:
#   ./encode-to-16k.sh [dir]           Convert dir/*.mp3 -> dir/*-16k.f32
#   ./encode-to-16k.sh --reverse [dir] Convert dir/*.f32 -> dir/*.mp3
#
# TODO: Replace this script with a proper audio preprocessing API
# (e.g., the audio_decode RPC in github.com/accretional/ffmpeg-proto)
# so that audio conversion happens as part of the service pipeline
# rather than as a manual preprocessing step.

REVERSE=false
if [ "${1:-}" = "--reverse" ]; then
    REVERSE=true
    shift
fi

DATA_DIR="${1:-data}"

if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg is not installed."
    exit 1
fi

count=0

if [ "$REVERSE" = true ]; then
    for f32 in "$DATA_DIR"/*.f32; do
        [ -f "$f32" ] || continue
        out="${f32%.f32}.mp3"

        if [ -f "$out" ]; then
            echo "Already exists: $out"
        else
            echo "Converting: $f32 -> $out"
            ffmpeg -hide_banner -loglevel error \
                -f f32le -ar 16000 -ac 1 -i "$f32" \
                -b:a 128k "$out"
        fi
        count=$((count + 1))
    done
else
    for mp3 in "$DATA_DIR"/*.mp3; do
        [ -f "$mp3" ] || continue
        base="${mp3%.mp3}"
        out="${base}-16k.f32"

        if [ -f "$out" ]; then
            echo "Already exists: $out"
        else
            echo "Converting: $mp3 -> $out"
            ffmpeg -hide_banner -loglevel error -i "$mp3" \
                -ar 16000 -ac 1 -f f32le -acodec pcm_f32le "$out"
        fi
        count=$((count + 1))
    done
fi

echo "Processed $count file(s)."
