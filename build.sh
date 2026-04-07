#!/bin/bash
set -euo pipefail

IMAGE_NAME="vad"
CONTAINER_NAME="vad-run"
MAIN_PKG="${1:-./cmd/vad}"

echo "=== Building VAD container ==="
echo "MAIN_PKG: ${MAIN_PKG}"

docker build -t "$IMAGE_NAME" --build-arg "MAIN_PKG=${MAIN_PKG}" . 2>&1

echo ""
echo "=== Running container ==="

# Run the container, stop it after 3 seconds if it's still running
docker run --rm --name "$CONTAINER_NAME" -d "$IMAGE_NAME"

sleep 3

if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
    echo "Container still running, stopping..."
    docker stop "$CONTAINER_NAME" > /dev/null
    echo "Stopped."
else
    echo "Container exited on its own."
fi

# TODO: Add actual invocation checks once the server is implemented
# e.g., send a health check request, run a sample inference, etc.

echo ""
echo "Build and run complete."
