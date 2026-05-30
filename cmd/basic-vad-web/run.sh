#!/bin/bash
# basic-vad-web/run.sh
#
# Spin up one or more vad gRPC backends + the HTTP demo server in front.
#
# Usage:
#   ./run.sh                              # pyannote on :50051, demo on :8080
#   ./run.sh --all                        # all 4 working backends on :50051..:50054
#   ./run.sh --backends pyannote,silero   # custom subset
#   PORT=9000 ./run.sh                    # change demo HTTP port
#
# Requires ./bin/vad to already exist. Run ./build-native.sh first if not.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

HTTP_PORT="${PORT:-8080}"
VAD_BIN="$REPO_ROOT/bin/vad"

# Parse args.
BACKENDS=""
case "${1:-}" in
    --all)
        BACKENDS="pyannote,fsmn,firered,silero"
        shift || true
        ;;
    --backends)
        BACKENDS="$2"
        shift 2
        ;;
    *)
        # Default: one pyannote backend.
        BACKENDS="pyannote"
        ;;
esac

if [ ! -x "$VAD_BIN" ]; then
    echo "ERROR: $VAD_BIN not found. Run ./build-native.sh first." >&2
    exit 1
fi

# Build the demo HTTP server.
echo "=== Building basic-vad-web ==="
DEMO_BIN="$SCRIPT_DIR/basic-vad-web"
go build -o "$DEMO_BIN" ./cmd/basic-vad-web/

# Spawn one vad instance per backend, on consecutive ports starting at 50051.
PIDS=()
ADDRS=()
BASE_PORT=50051
i=0
IFS=',' read -ra BACKEND_LIST <<< "$BACKENDS"
for b in "${BACKEND_LIST[@]}"; do
    b_trim="$(echo "$b" | xargs)"
    port=$((BASE_PORT + i))
    short="$(echo "$b_trim" | tr '[:lower:]' '[:upper:]')"
    echo "=== Starting vad backend $short on :$port ==="
    "$VAD_BIN" -backend "$b_trim" -port "$port" &
    PIDS+=($!)
    ADDRS+=("${short}=localhost:${port}")
    i=$((i + 1))
done

cleanup() {
    echo ""
    echo "Stopping demo + backends..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    [ -n "${DEMO_PID:-}" ] && kill "$DEMO_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f "$DEMO_BIN"
}
trap cleanup EXIT

# Give the backends a moment to bind + load weights.
echo "Waiting for backends to come up..."
for addr_spec in "${ADDRS[@]}"; do
    addr="${addr_spec#*=}"
    host="${addr%%:*}"
    port="${addr##*:}"
    for attempt in $(seq 1 40); do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "  $addr_spec ready"
            break
        fi
        sleep 0.25
        if [ "$attempt" = "40" ]; then
            echo "  $addr_spec NEVER CAME UP — check vad logs above" >&2
        fi
    done
done

# Join addresses for -vad-addrs.
VAD_ADDRS=$(IFS=','; echo "${ADDRS[*]}")

echo ""
echo "=== Starting demo HTTP server on :$HTTP_PORT ==="
echo "    -vad-addrs $VAD_ADDRS"
"$DEMO_BIN" -port "$HTTP_PORT" -vad-addrs "$VAD_ADDRS" &
DEMO_PID=$!

# Wait for demo to be ready.
URL="http://localhost:${HTTP_PORT}"
for attempt in $(seq 1 30); do
    if curl -s -o /dev/null "$URL/"; then
        echo "Demo ready: $URL"
        break
    fi
    sleep 0.25
done

# Open browser unless CI.
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
