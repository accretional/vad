#!/usr/bin/env bash
#
# Regenerate Go gRPC client + server bindings from proto/vad.proto.
#
# Outputs (overwritten in place):
#   proto/vadpb/vad.pb.go         message types
#   proto/vadpb/vad_grpc.pb.go    client + server interfaces
#
# Idempotent — protoc writes the same bytes given the same input + plugin
# versions, so running this on a clean tree should leave no diff. Used at
# the top of release.sh so a stale generated file fails the clean-tree
# gate before anything else builds.
#
# Requires protoc + protoc-gen-go + protoc-gen-go-grpc. setup.sh prints
# install hints if any are missing.
#
# Usage:
#   bash scripts/gen-proto.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

for tool in protoc protoc-gen-go protoc-gen-go-grpc; do
    if ! command -v "$tool" >/dev/null; then
        echo "ERROR: $tool not in PATH. Run bash setup.sh for install hints." >&2
        exit 1
    fi
done

mkdir -p proto/vadpb
protoc \
    --go_out=proto/vadpb --go_opt=paths=source_relative \
    --go-grpc_out=proto/vadpb --go-grpc_opt=paths=source_relative \
    -I=proto \
    proto/vad.proto

echo "generated:"
ls -l proto/vadpb/*.pb.go | awk '{printf "  %s  %s\n", $5, $NF}'
