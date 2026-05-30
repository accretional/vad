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
#   bash scripts/build_clients.sh
#
# Future-extension hook: add Python / C++ / Java client generation here as
# additional protoc plugins. For now Go is the only client we ship.
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

echo "generated (Go):"
ls -l proto/vadpb/*.pb.go | awk '{printf "  %s  %s\n", $5, $NF}'

# Python client (optional — only if grpcio-tools is installed).
# Drops into proto/vadpy/{vad_pb2.py, vad_pb2_grpc.py}. Clients can
# then `from vadpy import vad_pb2, vad_pb2_grpc`.
if command -v python3 >/dev/null 2>&1 && python3 -c "import grpc_tools" 2>/dev/null; then
    mkdir -p proto/vadpy
    python3 -m grpc_tools.protoc \
        -I=proto \
        --python_out=proto/vadpy \
        --grpc_python_out=proto/vadpy \
        proto/vad.proto
    echo "generated (Python):"
    ls -l proto/vadpy/*.py | awk '{printf "  %s  %s\n", $5, $NF}'
else
    echo "(Python client skipped: pip install grpcio-tools to enable)"
fi

# TODO: also generate C++ and Java clients when we have downstream
# consumers that need them. Each is one additional --<lang>_out plugin.
