#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-8080}"

# --- Detect ORT library ---

if [ -z "${ONNXRUNTIME_LIB:-}" ]; then
    ARCH=$(uname -m)
    OS=$(uname -s)
    if [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then PLATFORM="osx-arm64"; else PLATFORM="osx-x86_64"; fi
        ORT_LIB="libonnxruntime.dylib"
    else
        if [ "$ARCH" = "x86_64" ]; then PLATFORM="linux-x64"; else PLATFORM="linux-aarch64"; fi
        ORT_LIB="libonnxruntime.so"
    fi
    ORT_DIR="third_party/onnxruntime-${PLATFORM}-1.22.0/lib"
    if [ -d "$ORT_DIR" ]; then
        ORT_ABS="$(cd "$ORT_DIR" && pwd)"
        export ONNXRUNTIME_LIB="${ORT_ABS}/${ORT_LIB}"
        if [ "$OS" = "Darwin" ]; then
            export DYLD_LIBRARY_PATH="${ORT_ABS}:${DYLD_LIBRARY_PATH:-}"
        else
            export LD_LIBRARY_PATH="${ORT_ABS}:${LD_LIBRARY_PATH:-}"
        fi
    else
        echo "ERROR: ORT not found at $ORT_DIR. Run ./setup.sh first."
        exit 1
    fi
fi

echo "Using ORT: $ONNXRUNTIME_LIB"

# --- Build ---

echo "=== Building basic-vad-web ==="
BIN="$SCRIPT_DIR/basic-vad-web"
go build -o "$BIN" ./cmd/basic-vad-web/

# --- Run ---

URL="http://localhost:${PORT}"

echo "=== Starting server on ${URL} ==="
"$BIN" -port "$PORT" &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    rm -f "$BIN"
}
trap cleanup EXIT

# Wait for server to be ready
for i in $(seq 1 15); do
    if curl -s -o /dev/null "$URL/" 2>/dev/null; then
        echo "Server is ready."
        break
    fi
    sleep 0.5
done

# --- Validate ---

echo ""
echo "=== Validating responses ==="

# Check index.html
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL/")
if [ "$STATUS" = "200" ]; then
    echo "  GET /           -> $STATUS OK"
else
    echo "  GET /           -> $STATUS FAIL"
    exit 1
fi

# Check CSS
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL/static/style.css")
if [ "$STATUS" = "200" ]; then
    echo "  GET /static/style.css -> $STATUS OK"
else
    echo "  GET /static/style.css -> $STATUS FAIL"
    exit 1
fi

# Check JS
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL/static/app.js")
if [ "$STATUS" = "200" ]; then
    echo "  GET /static/app.js   -> $STATUS OK"
else
    echo "  GET /static/app.js   -> $STATUS FAIL"
    exit 1
fi

# Check 404
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL/nonexistent")
if [ "$STATUS" = "404" ]; then
    echo "  GET /nonexistent     -> $STATUS OK (expected)"
else
    echo "  GET /nonexistent     -> $STATUS FAIL (expected 404)"
    exit 1
fi

echo ""
echo "=== All checks passed ==="

# --- Open browser ---

echo "Opening $URL in browser..."
if command -v open &>/dev/null; then
    open "$URL"
elif command -v xdg-open &>/dev/null; then
    xdg-open "$URL"
else
    echo "  (no browser opener found, visit $URL manually)"
fi

echo ""
echo "Press Ctrl+C to stop the server."
wait "$SERVER_PID"
