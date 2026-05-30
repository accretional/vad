#!/bin/bash
set -euo pipefail

echo "=== VAD Project Setup ==="

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    echo "  https://docs.docker.com/get-docker/"
    exit 1
fi

echo "Docker found: $(docker --version)"

# Check Docker daemon is running
if ! docker info &> /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Please start Docker."
    exit 1
fi

echo "Docker daemon is running."

# Download ONNX weights for pyannote. The binary ships with these embedded via
# go:embed (see internal/embedded/), so this step is only required for users
# who want to override / update without rebuilding. Same goes for the alternate
# backends — fsmn-vad, firered-vad, marblenet, silero — whose weights are
# committed under weights/<backend>/ and bundled at build time.
WEIGHTS_DIR="weights"
MODEL_FILE="${WEIGHTS_DIR}/pyannote/model.onnx"
LEGACY_FILE="${WEIGHTS_DIR}/model.onnx"  # pre-2026-05 layout
MODEL_URL="https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx"

if [ -f "$MODEL_FILE" ]; then
    echo "Pyannote weights already at $MODEL_FILE"
elif [ -f "$LEGACY_FILE" ]; then
    echo "Migrating pyannote weights: $LEGACY_FILE → $MODEL_FILE"
    mkdir -p "$(dirname "$MODEL_FILE")"
    mv "$LEGACY_FILE" "$MODEL_FILE"
else
    echo "Downloading pyannote ONNX weights (~5.7 MB)..."
    mkdir -p "$(dirname "$MODEL_FILE")"
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
    echo "Downloaded weights to $MODEL_FILE"
fi

# Download ONNX Runtime shared library for local development
ORT_VERSION="1.22.0"
ORT_DIR="third_party"

ARCH=$(uname -m)
OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        ORT_PLATFORM="osx-arm64"
    else
        ORT_PLATFORM="osx-x86_64"
    fi
    ORT_LIB="libonnxruntime.dylib"
elif [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        ORT_PLATFORM="linux-x64"
    elif [ "$ARCH" = "aarch64" ]; then
        ORT_PLATFORM="linux-aarch64"
    else
        echo "ERROR: Unsupported Linux architecture: $ARCH"
        exit 1
    fi
    ORT_LIB="libonnxruntime.so"
else
    echo "ERROR: Unsupported OS: $OS"
    exit 1
fi

ORT_DIRNAME="onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}"
ORT_ARCHIVE="${ORT_DIRNAME}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"
ORT_LIB_PATH="${ORT_DIR}/${ORT_DIRNAME}/lib/${ORT_LIB}"

if [ -f "$ORT_LIB_PATH" ]; then
    echo "ONNX Runtime already downloaded: $ORT_LIB_PATH"
else
    echo "Downloading ONNX Runtime v${ORT_VERSION} for ${ORT_PLATFORM}..."
    mkdir -p "$ORT_DIR"
    curl -sL -o "${ORT_DIR}/${ORT_ARCHIVE}" "$ORT_URL"
    tar xzf "${ORT_DIR}/${ORT_ARCHIVE}" -C "$ORT_DIR"
    rm -f "${ORT_DIR}/${ORT_ARCHIVE}"
    echo "Downloaded ONNX Runtime to ${ORT_DIR}/${ORT_DIRNAME}/"
fi

# Check for protoc and Go gRPC plugins (needed to regenerate proto files)
echo ""
PROTO_OK=true
if ! command -v protoc &> /dev/null; then
    echo "WARNING: protoc is not installed. Needed to regenerate .proto files."
    echo "  macOS: brew install protobuf"
    echo "  Linux: apt install -y protobuf-compiler"
    PROTO_OK=false
else
    echo "protoc found: $(protoc --version)"
fi

if ! command -v protoc-gen-go &> /dev/null; then
    echo "WARNING: protoc-gen-go not installed."
    echo "  go install google.golang.org/protobuf/cmd/protoc-gen-go@latest"
    PROTO_OK=false
else
    echo "protoc-gen-go found: $(which protoc-gen-go)"
fi

if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "WARNING: protoc-gen-go-grpc not installed."
    echo "  go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"
    PROTO_OK=false
else
    echo "protoc-gen-go-grpc found: $(which protoc-gen-go-grpc)"
fi

if [ "$PROTO_OK" = true ]; then
    echo "All proto tools available. Regenerate with:"
    echo "  protoc --go_out=proto/vadpb --go_opt=paths=source_relative \\"
    echo "    --go-grpc_out=proto/vadpb --go-grpc_opt=paths=source_relative \\"
    echo "    -I=proto proto/vad.proto"
fi

# Check for ffmpeg (needed for audio encoding)
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "WARNING: ffmpeg is not installed. Needed for audio encoding."
    echo "  macOS: brew install ffmpeg"
    echo "  Linux: apt install -y ffmpeg"
else
    echo "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
fi

# Print environment setup hint
ORT_LIB_DIR="$(cd "${ORT_DIR}/${ORT_DIRNAME}/lib" && pwd)"
echo ""
echo "To use ONNX Runtime locally, set:"
if [ "$OS" = "Darwin" ]; then
    echo "  export DYLD_LIBRARY_PATH=${ORT_LIB_DIR}:\$DYLD_LIBRARY_PATH"
    echo "  export ONNXRUNTIME_LIB=${ORT_LIB_DIR}/${ORT_LIB}"
else
    echo "  export LD_LIBRARY_PATH=${ORT_LIB_DIR}:\$LD_LIBRARY_PATH"
    echo "  export ONNXRUNTIME_LIB=${ORT_LIB_DIR}/${ORT_LIB}"
fi

# Verify alternate-backend weights are present (they're bundled in git, not
# downloaded; see .gitignore notes for rationale).
echo ""
for backend in fsmn-vad firered-vad; do
    if [ -f "weights/${backend}/model.onnx" ]; then
        echo "Backend ${backend} weights present: weights/${backend}/"
    else
        echo "WARNING: weights/${backend}/model.onnx missing. The -backend ${backend%-vad} flag won't work."
        echo "  Either pull the latest commit (weights are bundled), or regenerate via:"
        echo "    https://github.com/accretional/speax/blob/main/benchmarks/vad/export_${backend//-/_}_to_onnx.py"
    fi
done

echo ""
echo "Setup complete. Build either way:"
echo "  ./build-native.sh        # local Go build; produces ./bin/vad (~59 MB self-contained)"
echo "  ./build.sh               # Docker image"
echo ""
echo "Available VAD backends (-backend or VADConfig.model in -config):"
echo "  pyannote   default; full diarization + speaker IDs (~6 MB)"
echo "  fsmn       tiny FunASR FSMN-VAD (~1.6 MB); VAD-only"
echo "  firered    FireRed DFSMN-VAD (~2.5 MB); VAD-only"
echo "  silero     Silero VAD (~1.3 MB); VAD-only"
echo "  marblenet  NVIDIA MarbleNet (~1 MB); VAD-only; NeMo log-mel features"
