#!/usr/bin/env bash
# Generate Python stubs from proto/vad.proto into /tmp and run the streaming
# validation client against a running vad server.
#
#   bash tests/stream/run.sh                 # use bundled data/sorry-dave-16k.wav (if present)
#   bash tests/stream/run.sh /path/audio.wav # use specific audio
#   bash tests/stream/run.sh /path/audio.wav --addr remote-host:50051 --chunk-ms 50
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Pick audio: arg1 if given, else look in data/
AUDIO="${1:-}"
if [[ -z "$AUDIO" ]]; then
  for candidate in data/sorry-dave-16k.wav data/sorry-dave.wav data/*.wav; do
    if [[ -f "$candidate" ]]; then AUDIO="$candidate"; break; fi
  done
fi
if [[ -z "$AUDIO" || ! -f "$AUDIO" ]]; then
  echo "ERROR: no audio file found. Pass a wav path as the first arg, or place one under data/." >&2
  exit 1
fi
shift 2>/dev/null || true   # remove arg1 if present

# Build stubs in a scratch dir each run so we don't commit generated code.
STUB_DIR="$(mktemp -d)"
trap 'rm -rf "$STUB_DIR"' EXIT

PY="${PYTHON:-python3}"

echo "==> generating Python stubs to $STUB_DIR"
"$PY" -m grpc_tools.protoc \
  --proto_path="$REPO_ROOT/proto" \
  --python_out="$STUB_DIR" \
  --grpc_python_out="$STUB_DIR" \
  vad.proto

echo "==> running stream client against ${ADDR:-localhost:50051}"
exec env PYTHONPATH="$STUB_DIR" "$PY" "$REPO_ROOT/tests/stream/stream_client.py" "$AUDIO" "$@"
