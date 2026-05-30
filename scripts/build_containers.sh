#!/usr/bin/env bash
#
# Build the slim per-arch container images AND extract the fat per-arch
# standalone Linux binaries from the same Dockerfile (two separate buildx
# invocations per arch: --target runtime --load for the container,
# --target export --output type=local for the host binary).
#
# Outputs:
#   image: ${IMAGE_NAME}:amd64     loaded into local docker daemon
#   image: ${IMAGE_NAME}:arm64     loaded into local docker daemon
#   file:  out/amd64/vad           ELF linux/amd64, fully self-contained
#   file:  out/arm64/vad           ELF linux/arm64, fully self-contained
#
# Requires buildx with a docker-container builder (auto-creates one named
# vad-builder if absent). Multi-arch emulation needs colima with
# `--vz-rosetta` on Apple Silicon (qemu falls over on Go's amd64 atomics).
#
# Usage:
#   bash scripts/build_containers.sh                    # both arches
#   ARCHES="linux/amd64" bash scripts/build_containers.sh   # one arch
#
# Environment:
#   IMAGE_NAME=vad                  base name for the container tags
#   ARCHES="linux/amd64 linux/arm64"  space-separated platforms
#   BUILDER_NAME=vad-builder        buildx builder instance name
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="${IMAGE_NAME:-vad}"
ARCHES="${ARCHES:-linux/amd64 linux/arm64}"
BUILDER_NAME="${BUILDER_NAME:-vad-builder}"

# buildx setup — idempotent: reuse the builder if it exists.
if docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    docker buildx use "$BUILDER_NAME"
else
    docker buildx create --name "$BUILDER_NAME" --driver docker-container --use
fi
docker buildx inspect --bootstrap >/dev/null

rm -rf out
mkdir -p out

for arch in $ARCHES; do
    short=$(echo "$arch" | sed 's,linux/,,')

    echo ""
    echo "=== Slim container ${IMAGE_NAME}:${short} ($arch) ==="
    docker buildx build \
        --platform "$arch" \
        --target runtime \
        -t "${IMAGE_NAME}:${short}" \
        --load \
        .

    echo ""
    echo "=== Fat standalone binary out/${short}/vad ($arch) ==="
    mkdir -p "out/${short}"
    docker buildx build \
        --platform "$arch" \
        --target export \
        --output "type=local,dest=out/${short}" \
        .
    chmod +x "out/${short}/vad"
    echo "  built: out/${short}/vad  ($(du -h out/${short}/vad | awk '{print $1}'))"
done

echo ""
echo "=== Built images ==="
docker images "${IMAGE_NAME}" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"
echo ""
echo "=== Built standalone binaries ==="
for f in out/*/vad; do
    [ -f "$f" ] && echo "  $f  $(du -h "$f" | awk '{print $1}')"
done
