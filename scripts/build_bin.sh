#!/usr/bin/env bash
#
# Build the native vad binary for the host OS/arch. Wraps build-native.sh
# (which itself runs prep-embed.sh + go build). Pulled out as a release
# pipeline step so it can be invoked standalone for fast iteration on the
# binary itself.
#
# Output: ./bin/vad — self-contained, embeds the ORT dylib + every backend's
# weights. ~59 MB on darwin/arm64.
#
# Usage:
#   bash scripts/build_bin.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

bash build-native.sh
