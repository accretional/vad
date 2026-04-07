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

echo ""
echo "Setup complete. You can now build with:"
echo "  docker build -t vad ."
