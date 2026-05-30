#!/usr/bin/env bash
# Materialise the libonnxruntime shared lib + weights/ tree into
# internal/embedded/{ort,weights}/ so go:embed can pick them up.
#
# Called by build-native.sh before `go build`, and by anything that runs
# `go test` against the internal/embedded package. Safe to run multiple
# times — only copies if the source is newer than the destination.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

EMBED_ORT="internal/embedded/ort"
EMBED_WEIGHTS="internal/embedded/weights"

# 1. ORT shared library
OS=$(uname -s)
ARCH=$(uname -m)
case "$OS-$ARCH" in
    Darwin-arm64) PLATFORM="osx-arm64"; LIB="libonnxruntime.dylib" ;;
    Darwin-x86_64) PLATFORM="osx-x86_64"; LIB="libonnxruntime.dylib" ;;
    Linux-x86_64) PLATFORM="linux-x64";   LIB="libonnxruntime.so" ;;
    Linux-aarch64) PLATFORM="linux-aarch64"; LIB="libonnxruntime.so" ;;
    *) echo "ERROR: unsupported $OS-$ARCH for embed prep" >&2; exit 1 ;;
esac

ORT_SRC=$(ls -d third_party/onnxruntime-${PLATFORM}-*/lib/${LIB} 2>/dev/null | head -1)
if [[ -z "$ORT_SRC" ]]; then
    echo "ERROR: ORT lib not found under third_party/onnxruntime-${PLATFORM}-*/lib/${LIB}" >&2
    echo "  Run setup.sh to download it." >&2
    exit 1
fi
# Resolve any symlinks so embed gets the real file.
ORT_REAL=$(readlink -f "$ORT_SRC" 2>/dev/null || python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$ORT_SRC")
mkdir -p "$EMBED_ORT"
ORT_DST="$EMBED_ORT/$LIB"
if [[ ! -f "$ORT_DST" ]] || [[ "$ORT_REAL" -nt "$ORT_DST" ]]; then
    cp "$ORT_REAL" "$ORT_DST"
    echo "embed-prep: copied $ORT_REAL → $ORT_DST  ($(du -h "$ORT_DST" | awk '{print $1}'))"
else
    echo "embed-prep: $ORT_DST already up to date"
fi

# 2. Weights tree — mirror every backend dir from weights/ into the embed root.
mkdir -p "$EMBED_WEIGHTS"
# Use rsync if available (does timestamp + content checks); else cp -R.
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude='.DS_Store' weights/ "$EMBED_WEIGHTS/"
else
    rm -rf "$EMBED_WEIGHTS"/*
    cp -R weights/* "$EMBED_WEIGHTS/" 2>/dev/null || true
fi
echo "embed-prep: weights mirrored from weights/ to $EMBED_WEIGHTS/"
echo "  $(find "$EMBED_WEIGHTS" -name model.onnx | wc -l | tr -d ' ') model.onnx files embedded"
