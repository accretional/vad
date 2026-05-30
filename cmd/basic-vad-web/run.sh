#!/bin/bash
# basic-vad-web/run.sh
#
# Spin up ONE vad gRPC backend + the HTTP demo server in front. Batch
# inference now runs in the browser (onnxruntime-web), so the server-side
# backend is only used by:
#   - /fetch (proxies the Fetch RPC so the browser can download models /
#     get a url.txt redirect)
#   - /aux/<dir>/<file> (sidecars served straight from the embedded weights
#     tree — no RPC needed)
#   - /socket (live-streaming panel: bridges the mic to DetectStream
#     against the loaded server-side backend)
#
# Usage:
#   ./run.sh                              # pyannote on :50051, demo on :8080
#   ./run.sh --backend silero             # any single backend
#   PORT=9000 ./run.sh                    # change demo HTTP port
#   VAD_PORT=50055 ./run.sh               # change vad gRPC port
#
# Requires ./bin/vad to already exist. Run ./build-native.sh first if not.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

HTTP_PORT="${PORT:-8080}"
VAD_PORT="${VAD_PORT:-50051}"
VAD_BIN="$REPO_ROOT/bin/vad"

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

echo "=== Building basic-vad-web ==="
DEMO_BIN="$SCRIPT_DIR/basic-vad-web"
go build -o "$DEMO_BIN" ./cmd/basic-vad-web/

echo "=== Starting vad backend $BACKEND on :$VAD_PORT ==="
"$VAD_BIN" -backend "$BACKEND" -port "$VAD_PORT" &
VAD_PID=$!

cleanup() {
    echo ""
    echo "Stopping demo + vad..."
    [ -n "${DEMO_PID:-}" ] && kill "$DEMO_PID" 2>/dev/null || true
    kill "$VAD_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f "$DEMO_BIN"
}
trap cleanup EXIT

echo "Waiting for vad to come up..."
for attempt in $(seq 1 40); do
    if nc -z localhost "$VAD_PORT" 2>/dev/null; then
        echo "  vad backend ready on localhost:$VAD_PORT"
        break
    fi
    sleep 0.25
    if [ "$attempt" = "40" ]; then
        echo "  vad NEVER CAME UP — check logs above" >&2
    fi
done

echo ""
echo "=== Starting demo HTTP server on :$HTTP_PORT ==="
echo "    -vad-addr localhost:$VAD_PORT"
"$DEMO_BIN" -port "$HTTP_PORT" -vad-addr "localhost:$VAD_PORT" &
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
