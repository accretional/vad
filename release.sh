#!/usr/bin/env bash
#
# Release builder. Produces every shippable artifact for the vad project:
#
#   bin/vad                           native binary for the host triple
#                                     (darwin/arm64, linux/amd64, ...)
#   vad:linux-amd64                   linux/amd64 container
#   vad:linux-arm64                   linux/arm64 container
#   vad:<tag> + vad:latest manifest   multi-arch manifest list referencing
#                                     both arches (only when --push is given —
#                                     manifest lists can't be loaded into the
#                                     local daemon, only pushed to a registry)
#
# Pipeline:
#   1. Sanity: clean tree, on a known branch, all tests green.
#   2. Native build via build-native.sh (host arch only).
#   3. buildx ensure: a docker-container builder named vad-builder.
#   4. Per-arch container build with --load (so the local daemon has both).
#   5. (--push only) Push to ${REGISTRY:-ghcr.io/accretional/vad} as a
#      multi-arch manifest.
#   6. (--release only) git tag + gh release create + upload bin/vad.
#
# Usage:
#   bash release.sh                       # build all artifacts locally
#   bash release.sh --tag v0.2.0          # version tag for the manifest + GH release
#   bash release.sh --skip-tests          # skip the test.sh step (fast iteration)
#   bash release.sh --push                # additionally push images to $REGISTRY
#   bash release.sh --release --tag v0.2.0
#                                         # additionally git-tag + gh release create
#
# Environment:
#   REGISTRY=ghcr.io/accretional/vad      # image repo for --push / --release (defaults shown)
#   ARCHES="linux/amd64 linux/arm64"      # which arches to build (override for single-arch)
#   SKIP_AMD64=1 / SKIP_ARM64=1           # alternate per-arch skip switches
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="vad"
REGISTRY="${REGISTRY:-ghcr.io/accretional/vad}"
BUILDER_NAME="vad-builder"
ARCHES="${ARCHES:-linux/amd64 linux/arm64}"

TAG=""
PUSH=0
RELEASE=0
SKIP_TESTS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        --push) PUSH=1; shift ;;
        --release) RELEASE=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | grep -E '^#' | sed 's/^# //;s/^#$//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

step() { echo ""; echo "=== $* ==="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ---- 1. Sanity ------------------------------------------------------------
step "Sanity checks"

if [ "$RELEASE" = 1 ] && [ -z "$TAG" ]; then
    die "--release requires --tag <vX.Y.Z>"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "working tree is dirty (commit or stash first):
$(git status --short)"
fi

GIT_SHA=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "  branch: $GIT_BRANCH  commit: $GIT_SHA  tag: ${TAG:-<none>}"

if [ "$SKIP_TESTS" = 0 ]; then
    step "Full test suite (test.sh)"
    bash test.sh
else
    echo "  --> SKIP: tests bypassed (--skip-tests)"
fi

# ---- 2. Native binary -----------------------------------------------------
step "Native build (host arch)"
bash build-native.sh
echo "  built: $(file bin/vad)"
echo "  size:  $(du -h bin/vad | awk '{print $1}')"

# ---- 3. buildx setup ------------------------------------------------------
step "buildx (docker-container driver)"
if docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    docker buildx use "$BUILDER_NAME"
else
    docker buildx create --name "$BUILDER_NAME" --driver docker-container --use
fi
docker buildx inspect --bootstrap >/dev/null

# ---- 4. Per-arch container + host-binary export --------------------------
# Two artifacts per architecture:
#   - runtime container (slim binary + /onnx tree) → loaded into local daemon
#   - fat standalone Linux binary → written to out/<arch>/vad on the host
rm -rf out
mkdir -p out
for arch in $ARCHES; do
    short=$(echo "$arch" | sed 's,linux/,,')
    case "$short" in
        amd64) [ "${SKIP_AMD64:-0}" = 1 ] && { echo "skipping amd64"; continue; } ;;
        arm64) [ "${SKIP_ARM64:-0}" = 1 ] && { echo "skipping arm64"; continue; } ;;
    esac

    step "Slim container: $arch  →  ${IMAGE_NAME}:${short}"
    docker buildx build \
        --platform "$arch" \
        --target runtime \
        -t "${IMAGE_NAME}:${short}" \
        --load \
        .
    echo "  built: ${IMAGE_NAME}:${short}  $(docker images --format '{{.Size}}' ${IMAGE_NAME}:${short})"

    step "Fat standalone binary: $arch  →  out/${short}/vad"
    mkdir -p "out/${short}"
    docker buildx build \
        --platform "$arch" \
        --target export \
        --output "type=local,dest=out/${short}" \
        .
    chmod +x "out/${short}/vad"
    echo "  built: out/${short}/vad  ($(du -h out/${short}/vad | awk '{print $1}'))"
done

# ---- 5. Multi-arch manifest (push only) -----------------------------------
if [ "$PUSH" = 1 ]; then
    step "Push multi-arch manifest to $REGISTRY"
    MANIFEST_TAG="${REGISTRY}:${TAG:-$GIT_SHA}"
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
fi

# ---- 6. GitHub release ----------------------------------------------------
if [ "$RELEASE" = 1 ]; then
    step "GitHub release ${TAG}"
    if git rev-parse "$TAG" >/dev/null 2>&1; then
        echo "  tag $TAG already exists; skipping git tag"
    else
        git tag -a "$TAG" -m "Release $TAG"
        git push origin "$TAG"
    fi
    # Cross-platform sha256
    if command -v sha256sum >/dev/null; then sha256sum bin/vad > bin/vad.sha256
    else shasum -a 256 bin/vad > bin/vad.sha256; fi
    if command -v gh >/dev/null; then
        gh release create "$TAG" \
            --title "$TAG" \
            --generate-notes \
            bin/vad bin/vad.sha256
        echo "  created: gh release $TAG"
    else
        echo "  gh CLI not installed — skipping release creation."
        echo "  Manual upload: bin/vad ($(du -h bin/vad | awk '{print $1}')), bin/vad.sha256"
    fi
fi

# ---- summary --------------------------------------------------------------
echo ""
echo "=== Release artifacts ==="
echo "Native (this host):"
echo "  bin/vad                       $(du -h bin/vad | awk '{print $1}')   $(file -b bin/vad | head -c 60)"
echo "Standalone Linux binaries (in out/):"
for f in out/*/vad; do
    [ -f "$f" ] && echo "  $f                $(du -h $f | awk '{print $1}')   $(file -b $f | head -c 60)"
done
echo "Slim runtime containers (local daemon):"
docker images "${IMAGE_NAME}" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}" | head
if [ "$PUSH" = 1 ]; then
    echo "Pushed:"
    echo "  $MANIFEST_TAG"
    echo "  $LATEST_TAG"
fi
echo ""
echo "OK"
