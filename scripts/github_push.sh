#!/usr/bin/env bash
#
# Push the release: multi-arch container manifest to $REGISTRY, then
# git tag, then `gh release create` with the zip + raw binaries.
#
# Release notes use `--generate-notes` (PR titles since the previous tag);
# no separate CHANGELOG file required.
#
# Required env:
#   TAG       version tag (e.g. v0.2.0); used for git tag + image tags
#   ZIP       path to the release zip created by release.sh
#
# Optional env:
#   REGISTRY  default: ghcr.io/accretional/vad
#   ARCHES    default: "linux/amd64 linux/arm64"
#   BUILDER_NAME  default: vad-builder
#
# Usage:
#   TAG=v0.2.0 ZIP=release_v0.2.0_abc1234.zip bash scripts/github_push.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

: "${TAG:?must set TAG=vX.Y.Z}"
: "${ZIP:?must set ZIP=release_*.zip}"
REGISTRY="${REGISTRY:-ghcr.io/accretional/vad}"
ARCHES="${ARCHES:-linux/amd64 linux/arm64}"
BUILDER_NAME="${BUILDER_NAME:-vad-builder}"

if [ ! -f "$ZIP" ]; then
    echo "ERROR: $ZIP missing — was the release zip built?" >&2
    exit 1
fi

# Dev-only safety knob: set RELEASE_DRY_RUN=1 to log every push command
# without actually invoking it. Lets you exercise release.sh end-to-end
# locally without creating real git tags or touching the registry. NOT a
# "--skip-*" flag (every step still runs); this only changes the mode
# of the push step.
if [ "${RELEASE_DRY_RUN:-0}" = 1 ]; then
    echo "=== DRY RUN — would push to $REGISTRY with tag $TAG, zip $ZIP ==="
    echo "  docker buildx build --platform $(echo "$ARCHES" | tr ' ' ',') --target runtime \\"
    echo "                      -t ${REGISTRY}:${TAG} -t ${REGISTRY}:latest --push ."
    echo "  git tag -a $TAG -m 'Release $TAG'"
    echo "  git push origin $TAG"
    echo "  gh release create $TAG --title $TAG --generate-notes $ZIP bin/vad out/amd64/vad out/arm64/vad"
    exit 0
fi

# ---- multi-arch image push ---------------------------------------------
echo ""
echo "=== push multi-arch manifest to $REGISTRY ==="
if docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    docker buildx use "$BUILDER_NAME"
else
    docker buildx create --name "$BUILDER_NAME" --driver docker-container --use
fi

MANIFEST_TAG="${REGISTRY}:${TAG}"
LATEST_TAG="${REGISTRY}:latest"
docker buildx build \
    --platform "$(echo "$ARCHES" | tr ' ' ',')" \
    --target runtime \
    -t "$MANIFEST_TAG" \
    -t "$LATEST_TAG" \
    --push \
    .
echo "  pushed: $MANIFEST_TAG"
echo "  pushed: $LATEST_TAG"

# TODO: cosign sign goes HERE:
#   cosign sign --yes "$MANIFEST_TAG"
#   cosign sign --yes "$LATEST_TAG"

# ---- git tag -----------------------------------------------------------
echo ""
echo "=== git tag $TAG ==="
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "  tag $TAG already exists; skipping"
else
    git tag -a "$TAG" -m "Release $TAG"
    git push origin "$TAG"
fi

# ---- gh release --------------------------------------------------------
echo ""
echo "=== gh release create $TAG ==="
if ! command -v gh >/dev/null; then
    echo "  gh CLI not installed — skipping. Manual upload required:"
    echo "    $ZIP"
    for f in bin/vad out/amd64/vad out/arm64/vad; do [ -f "$f" ] && echo "    $f"; done
    exit 0
fi

UPLOADS=("$ZIP")
for f in bin/vad out/amd64/vad out/arm64/vad; do
    [ -f "$f" ] && UPLOADS+=("$f")
done
gh release create "$TAG" \
    --title "$TAG" \
    --generate-notes \
    "${UPLOADS[@]}"
echo "  created: gh release $TAG"
