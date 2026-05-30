#!/usr/bin/env bash
# Native (non-Docker) build of the vad gRPC server. Materialises the embed
# artifacts via prep-embed.sh, then `go build`.
#
# Output: ./bin/vad — a self-contained binary that ships with libonnxruntime
# and all on-disk backend weights bundled via go:embed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

bash prep-embed.sh

mkdir -p bin
echo "=== go build ==="
go build -o bin/vad ./cmd/vad
echo "built: bin/vad ($(du -h bin/vad | awk '{print $1}'))"
echo ""
echo "Start with:"
echo "  ./bin/vad                                  # pyannote (default)"
echo "  ./bin/vad -backend fsmn                    # FSMN-VAD"
echo "  ./bin/vad -backend firered                 # FireRedVAD"
echo "  ./bin/vad -backend silero                  # Silero VAD"
echo "  ./bin/vad -config configs/example.textproto"
