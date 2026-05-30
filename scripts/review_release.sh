#!/usr/bin/env bash
#
# Open the HTML release-review page in the operator's browser and block
# until they click ACCEPT or CANCEL. Wraps the release-review Go binary,
# which serves a tiny HTTP server backed by metadata.json + the per-step
# logs from the supplied --log-dir.
#
# Exit code:
#   0   operator clicked ACCEPT
#   1   operator clicked CANCEL (release.sh should bail; nothing pushed)
#
# Usage:
#   bash scripts/review_release.sh release-logs/v0.2.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${1:-}"
if [ -z "$LOG_DIR" ]; then
    echo "usage: $0 <log-dir>" >&2
    exit 2
fi
if [ ! -f "$LOG_DIR/metadata.json" ]; then
    echo "ERROR: $LOG_DIR/metadata.json missing — release.sh aggregates it before review" >&2
    exit 2
fi

# Build the review binary into the log dir itself so it gets zipped along
# with everything else (useful for post-mortems).
BIN="$LOG_DIR/_release-review"
go build -o "$BIN" ./cmd/release-review

"$BIN" -logs "$LOG_DIR"
