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

# Download ONNX weights
WEIGHTS_DIR="weights"
MODEL_FILE="${WEIGHTS_DIR}/model.onnx"
MODEL_URL="https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx"

if [ -f "$MODEL_FILE" ]; then
    echo "Weights already downloaded: $MODEL_FILE"
else
    echo "Downloading ONNX weights..."
    mkdir -p "$WEIGHTS_DIR"
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

echo ""
echo "Setup complete. You can now build with:"
echo "  ./build.sh"
