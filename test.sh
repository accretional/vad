#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="vad"
TEST_CONTAINERS="vad-test-e2e vad-test-fetch-direct vad-test-fetch-url"
TEST_PORTS="50051 50052 50053 18080"

# --- Cleanup function ---

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    for name in $TEST_CONTAINERS; do
        docker rm -f "$name" 2>/dev/null || true
    done
    for port in $TEST_PORTS; do
        lsof -ti ":$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
}

# Always clean up on exit
trap cleanup EXIT

# --- Pre-run cleanup ---

cleanup

# --- Detect ORT for local tests ---

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
        echo "Using ORT: $ONNXRUNTIME_LIB"
    else
        echo "WARNING: ORT not found at $ORT_DIR — unit tests may be skipped"
    fi
fi

# --- Step 1: Go vet ---

echo ""
echo "=== go vet ==="
go vet ./...

# --- Step 2: Unit tests ---

echo ""
echo "=== Unit tests ==="
go test -v ./pkg/... ./cmd/basic-vad-web/

# --- Step 3: Docker build ---

echo ""
echo "=== Docker build ==="
docker build --no-cache -t "$IMAGE_NAME" --build-arg MAIN_PKG=./cmd/vad .

# --- Step 4: E2E tests (uses pre-built image) ---

echo ""
echo "=== E2E integration tests ==="
go test -v -timeout 120s ./tests/e2e/

# --- Step 5: Fetch tests (uses pre-built image) ---

echo ""
echo "=== Fetch integration tests ==="
go test -v -timeout 120s ./tests/fetch/

# --- Step 6: Basic VAD web tests (local server, no Docker) ---

echo ""
echo "=== Basic VAD web integration tests ==="
go test -v -timeout 60s ./tests/basic-vad-web/

# --- Done ---

echo ""
echo "=== All tests passed ==="
