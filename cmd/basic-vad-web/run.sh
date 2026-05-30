#!/bin/bash
# basic-vad-web/run.sh
#
# Spin up THREE processes: one vad gRPC backend, one speax/audio gRPC
# backend, and the HTTP demo server in front. Batch inference now runs in
# the browser (onnxruntime-web), so the server-side processes only handle:
#
#   vad (:50051)   — /fetch (model weights), /socket (DetectStream)
#   audio (:50052) — /upload (decode arbitrary container → 16 kHz mono WAV
#                    via speax/audio's MediaConverter), /svg (waveform SVG
#                    via AudioToVectors + VectorsToSvg)
#   demo (:8080)   — HTTP front, serves embedded UI + proxies both backends
#
# Usage:
#   ./run.sh                              # pyannote on :50051, audio on :50052, demo on :8080
#   ./run.sh --backend silero             # any single vad backend
#   PORT=9000 ./run.sh                    # change demo HTTP port
#   VAD_PORT=50055 ./run.sh               # change vad gRPC port
#   AUDIO_PORT=50066 ./run.sh             # change audio gRPC port
#   AUDIO_REPO=/path/to/speax/audio       # override sibling-checkout path
#
# Requires ./bin/vad to already exist. Run ./build-native.sh first if not.
# Also requires a sibling checkout of speax/audio (default: ../speax/audio
# from this repo's root) — the audio server binary is built from there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

HTTP_PORT="${PORT:-8080}"
VAD_PORT="${VAD_PORT:-50051}"
AUDIO_PORT="${AUDIO_PORT:-50052}"
VAD_BIN="$REPO_ROOT/bin/vad"
AUDIO_REPO="${AUDIO_REPO:-$REPO_ROOT/../speax/audio}"

BACKEND="pyannote"
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -x "$VAD_BIN" ]; then
    echo "ERROR: $VAD_BIN not found. Run ./build-native.sh first." >&2
    exit 1
fi

if [ ! -d "$AUDIO_REPO" ]; then
    echo "ERROR: speax/audio checkout not found at $AUDIO_REPO." >&2
    echo "       Set AUDIO_REPO=/path/to/speax/audio or clone it as a sibling of this repo." >&2
    exit 1
fi

echo "=== Building basic-vad-web ==="
DEMO_BIN="$SCRIPT_DIR/basic-vad-web"
go build -o "$DEMO_BIN" ./cmd/basic-vad-web/

echo "=== Building audio server (from $AUDIO_REPO) ==="
AUDIO_BIN="$SCRIPT_DIR/audio-server"
( cd "$AUDIO_REPO" && go build -o "$AUDIO_BIN" ./cmd/server )

echo "=== Starting vad backend $BACKEND on :$VAD_PORT ==="
"$VAD_BIN" -backend "$BACKEND" -port "$VAD_PORT" &
VAD_PID=$!

echo "=== Starting audio server on :$AUDIO_PORT ==="
"$AUDIO_BIN" -port "$AUDIO_PORT" &
AUDIO_PID=$!

cleanup() {
    echo ""
    echo "Stopping demo + vad + audio..."
    [ -n "${DEMO_PID:-}" ] && kill "$DEMO_PID" 2>/dev/null || true
    kill "$VAD_PID" 2>/dev/null || true
    kill "$AUDIO_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f "$DEMO_BIN" "$AUDIO_BIN"
}
trap cleanup EXIT

echo "Waiting for vad + audio to come up..."
for attempt in $(seq 1 40); do
    if nc -z localhost "$VAD_PORT" 2>/dev/null && nc -z localhost "$AUDIO_PORT" 2>/dev/null; then
        echo "  vad ready on localhost:$VAD_PORT"
        echo "  audio ready on localhost:$AUDIO_PORT"
        break
    fi
    sleep 0.25
    if [ "$attempt" = "40" ]; then
        echo "  one of vad/audio NEVER CAME UP — check logs above" >&2
    fi
done

echo ""
echo "=== Starting demo HTTP server on :$HTTP_PORT ==="
echo "    -vad-addr localhost:$VAD_PORT"
echo "    -audio-addr localhost:$AUDIO_PORT"
"$DEMO_BIN" -port "$HTTP_PORT" \
    -vad-addr "localhost:$VAD_PORT" \
    -audio-addr "localhost:$AUDIO_PORT" &
DEMO_PID=$!

URL="http://localhost:${HTTP_PORT}"
for attempt in $(seq 1 30); do
    if curl -s -o /dev/null "$URL/"; then
        echo "Demo ready: $URL"
        break
    fi
    sleep 0.25
done

if [ "${CI:-}" != "1" ]; then
    if command -v open &>/dev/null; then
        open "$URL"
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$URL"
    fi
fi

echo ""
echo "Press Ctrl+C to stop."
wait "$DEMO_PID"
