#!/bin/bash
#
# Full test suite. Steps that need Docker or ffmpeg are skipped (not failed)
# when those tools aren't available, so this script is safe to run on any
# dev box.
#
# Layers, fastest first:
#   1. go vet ./...
#   2. Unit tests for every Go package (pure Go + CGO-backed ORT).
#   3. tests/basic-vad-web (in-process HTTP server, no Docker).
#   4. Native pkg-example pipeline (needs ffmpeg + third_party/ ORT).
#   5. Docker build + tests/e2e (per-backend) + tests/fetch (needs `docker info` OK).
#
# Exits non-zero on any failure in a layer that actually ran. Skipped layers
# are summarised at the end.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="vad"
TEST_CONTAINERS="vad-test-e2e vad-test-fetch-direct vad-test-fetch-url"
TEST_PORTS="50051 50052 50053 18080"

# ---- helpers --------------------------------------------------------------

step() { echo ""; echo "=== $* ==="; }
warn() { echo "  --> SKIP: $*"; }

has_docker() { docker version --format '{{.Server.Version}}' >/dev/null 2>&1; }
has_ffmpeg() { command -v ffmpeg >/dev/null 2>&1; }

cleanup() {
    echo ""
    step "Cleaning up"
    if has_docker; then
        for name in $TEST_CONTAINERS vad-test-backend-pyannote vad-test-backend-fsmn \
                    vad-test-backend-firered vad-test-backend-silero vad-test-backend-marblenet; do
            docker rm -f "$name" 2>/dev/null || true
        done
    fi
    for port in $TEST_PORTS 55051 55052 55053 55054 55055; do
        lsof -ti ":$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
}
trap cleanup EXIT
cleanup  # pre-run cleanup of any stragglers

# ---- pick up ORT for local cgo tests --------------------------------------
# Unit tests in pkg/vad/ load the ORT dylib at init. The embedded path is
# preferred at runtime but only populated post-prep-embed; for plain `go test`
# point at whatever's under third_party/, otherwise fall back to the env var.

if [ -z "${ONNXRUNTIME_LIB:-}" ]; then
    ARCH=$(uname -m); OS=$(uname -s)
    case "$OS-$ARCH" in
        Darwin-arm64)  PLATFORM=osx-arm64;     ORT_LIB=libonnxruntime.dylib ;;
        Darwin-x86_64) PLATFORM=osx-x86_64;    ORT_LIB=libonnxruntime.dylib ;;
        Linux-x86_64)  PLATFORM=linux-x64;     ORT_LIB=libonnxruntime.so ;;
        Linux-aarch64) PLATFORM=linux-aarch64; ORT_LIB=libonnxruntime.so ;;
        *) PLATFORM=""; ORT_LIB="" ;;
    esac
    ORT_DIR="third_party/onnxruntime-${PLATFORM}-1.22.0/lib"
    if [ -n "$PLATFORM" ] && [ -d "$ORT_DIR" ]; then
        ORT_ABS="$(cd "$ORT_DIR" && pwd)"
        export ONNXRUNTIME_LIB="${ORT_ABS}/${ORT_LIB}"
        if [ "$OS" = "Darwin" ]; then
            export DYLD_LIBRARY_PATH="${ORT_ABS}:${DYLD_LIBRARY_PATH:-}"
        else
            export LD_LIBRARY_PATH="${ORT_ABS}:${LD_LIBRARY_PATH:-}"
        fi
        echo "Using ORT: $ONNXRUNTIME_LIB"
    else
        echo "NOTE: ORT not found under third_party/onnxruntime-${PLATFORM}-1.22.0/lib"
        echo "      Run bash setup.sh to download it. CGO-backed tests will fail without it."
    fi
fi

FAILED=0
RAN=0
SKIPPED=()

run_step() {
    local label="$1"; shift
    step "$label"
    if "$@"; then
        RAN=$((RAN+1))
    else
        echo "  --> FAIL: $label"
        FAILED=$((FAILED+1))
    fi
}

skip_step() {
    SKIPPED+=("$1")
    step "$1"
    warn "$2"
}

# ---- 1. go vet ------------------------------------------------------------
run_step "go vet ./..." go vet ./...

# ---- 2. Unit tests --------------------------------------------------------
run_step "Unit tests" \
    go test -timeout 120s ./fbank/... ./melspec/... ./pkg/... ./internal/... ./cmd/...

# ---- 3. tests/basic-vad-web (HTTP server in-process) ----------------------
run_step "basic-vad-web integration tests" \
    go test -timeout 60s ./tests/basic-vad-web/

# ---- 4. pkg-example native pipeline (needs ffmpeg) ------------------------
if has_ffmpeg; then
    run_step "cmd/pkg-example/run.sh" bash cmd/pkg-example/run.sh
else
    skip_step "cmd/pkg-example/run.sh" \
        "ffmpeg not installed — brew install ffmpeg (macOS) / apt install ffmpeg (Linux)"
fi

# ---- 5. Docker build + e2e + fetch tests ----------------------------------
if has_docker; then
    run_step "Docker build" \
        docker build --no-cache -t "$IMAGE_NAME" --build-arg MAIN_PKG=./cmd/vad .
    run_step "tests/e2e (pyannote default + all-backends matrix)" \
        go test -timeout 600s ./tests/e2e/
    run_step "tests/fetch (Docker)" \
        go test -timeout 180s ./tests/fetch/
else
    skip_step "Docker build + tests/e2e + tests/fetch" \
        "Docker daemon not reachable. On macOS w/ colima: colima stop && colima start --runtime docker"
fi

# ---- summary --------------------------------------------------------------
echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "=== ${FAILED} step(s) FAILED (${RAN} passed, ${#SKIPPED[@]} skipped) ==="
    exit 1
fi
echo "=== ${RAN} step(s) passed, ${#SKIPPED[@]} skipped ==="
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    for s in "${SKIPPED[@]}"; do echo "  skipped: $s"; done
fi
