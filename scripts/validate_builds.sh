#!/usr/bin/env bash
#
# Validate each built artifact (native binary, fat linux/<arch> binaries
# in scratch debian, slim per-arch containers) by standing each up on its
# own port and running the validate Go binary against it (Detect + Fetch
# for every backend in the proto enum).
#
# Each validation writes a per-artifact log into $LOG_DIR (default
# release-logs/_validate/) and the script exits non-zero if any artifact
# failed. Used by release.sh as the gate between build and review.
#
# Expects bin/vad + out/<arch>/vad + ${IMAGE_NAME}:<arch> to already exist
# (run build_bin.sh + build_containers.sh first).
#
# Usage:
#   bash scripts/validate_builds.sh
#   LOG_DIR=/tmp/vad-validate bash scripts/validate_builds.sh
#
# Environment:
#   LOG_DIR=release-logs/_validate       per-artifact log destination
#   IMAGE_NAME=vad                       slim container tag base
#   ARCHES="linux/amd64 linux/arm64"     which arch artifacts to validate
#
# Port allocation (high to dodge dev defaults):
#   50100  native (host bin)
#   50101  fat amd64 binary in debian-slim
#   50102  fat arm64 binary in debian-slim
#   50103  slim amd64 container
#   50104  slim arm64 container
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="${IMAGE_NAME:-vad}"
LOG_DIR="${LOG_DIR:-release-logs/_validate}"
ARCHES="${ARCHES:-linux/amd64 linux/arm64}"

PORT_NATIVE=50100
PORT_FAT_AMD64=50101
PORT_FAT_ARM64=50102
PORT_SLIM_AMD64=50103
PORT_SLIM_ARM64=50104

mkdir -p "$LOG_DIR"
VALIDATE_BIN="$LOG_DIR/_validate"
go build -o "$VALIDATE_BIN" ./tests/validate

CLEANUP_PIDS=()
CLEANUP_CONTAINERS=()
cleanup() {
    [ "${#CLEANUP_PIDS[@]}" -gt 0 ]       && kill "${CLEANUP_PIDS[@]}" 2>/dev/null || true
    [ "${#CLEANUP_CONTAINERS[@]}" -gt 0 ] && docker rm -f "${CLEANUP_CONTAINERS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

wait_port() {
    local p="$1" max="$2"
    for _ in $(seq 1 "$max"); do
        nc -z localhost "$p" 2>/dev/null && return 0
        sleep 1
    done
    return 1
}

# results.tsv: one row per validation — name<TAB>port<TAB>ok<TAB>log_path
RESULTS="$LOG_DIR/results.tsv"
: > "$RESULTS"
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RESULTS"; }

run_validate() {
    local name="$1" addr="$2" log="$3"
    "$VALIDATE_BIN" -addr "$addr" -wait 30s 2>&1 | tee "$log"
    return ${PIPESTATUS[0]}
}

# ---- native binary -------------------------------------------------------
if [ -x bin/vad ]; then
    name=native; port=$PORT_NATIVE; log="$LOG_DIR/${name}.log"
    echo ""
    echo "=== validate $name (port $port) ==="
    ./bin/vad -port "$port" > "$log.server" 2>&1 &
    pid=$!; CLEANUP_PIDS+=("$pid")
    if wait_port "$port" 30 && run_validate "$name" "localhost:$port" "$log"; then
        record "$name" "$port" true "${log#${LOG_DIR}/}"
    else
        record "$name" "$port" false "${log#${LOG_DIR}/}"
    fi
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
else
    echo "WARN: bin/vad missing — run scripts/build_bin.sh first"
fi

# ---- fat per-arch standalone binary in debian-slim -----------------------
for arch in $(echo "$ARCHES" | tr ' ' '\n' | sed 's,linux/,,'); do
    case "$arch" in
        amd64) port=$PORT_FAT_AMD64 ;;
        arm64) port=$PORT_FAT_ARM64 ;;
        *)     echo "skipping fat-$arch (no port allocated)"; continue ;;
    esac
    name="fat-$arch"; log="$LOG_DIR/${name}.log"
    src="out/${arch}/vad"
    if [ ! -x "$src" ]; then
        echo "WARN: $src missing — run scripts/build_containers.sh"
        continue
    fi
    echo ""
    echo "=== validate $name (port $port) ==="
    tmp=$(mktemp -d); cp "$src" "$tmp/"
    cat > "$tmp/Dockerfile" <<EOF
FROM debian:bookworm-slim
COPY vad /vad
ENTRYPOINT ["/vad"]
EOF
    img="vad-validate-${name}:test"
    docker build --platform "linux/${arch}" -t "$img" "$tmp" >> "$log.build" 2>&1
    rm -rf "$tmp"
    cname="vad-validate-${name}"
    docker run --rm -d --name "$cname" --platform "linux/${arch}" -p "$port:50051" "$img" >/dev/null 2>&1
    CLEANUP_CONTAINERS+=("$cname")
    if wait_port "$port" 30 && run_validate "$name" "localhost:$port" "$log"; then
        record "$name" "$port" true "${log#${LOG_DIR}/}"
    else
        docker logs "$cname" 2>&1 | tail -20 >> "$log"
        record "$name" "$port" false "${log#${LOG_DIR}/}"
    fi
    docker rm -f "$cname" >/dev/null 2>&1
    docker rmi "$img" >/dev/null 2>&1
done

# ---- slim per-arch container --------------------------------------------
for arch in $(echo "$ARCHES" | tr ' ' '\n' | sed 's,linux/,,'); do
    case "$arch" in
        amd64) port=$PORT_SLIM_AMD64 ;;
        arm64) port=$PORT_SLIM_ARM64 ;;
        *)     echo "skipping slim-$arch (no port allocated)"; continue ;;
    esac
    name="slim-$arch"; log="$LOG_DIR/${name}.log"
    img="${IMAGE_NAME}:${arch}"
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "WARN: image $img missing — run scripts/build_containers.sh"
        continue
    fi
    echo ""
    echo "=== validate $name (port $port) ==="
    cname="vad-validate-${name}"
    docker run --rm -d --name "$cname" --platform "linux/${arch}" -p "$port:50051" "$img" >/dev/null 2>&1
    CLEANUP_CONTAINERS+=("$cname")
    if wait_port "$port" 30 && run_validate "$name" "localhost:$port" "$log"; then
        record "$name" "$port" true "${log#${LOG_DIR}/}"
    else
        docker logs "$cname" 2>&1 | tail -20 >> "$log"
        record "$name" "$port" false "${log#${LOG_DIR}/}"
    fi
    docker rm -f "$cname" >/dev/null 2>&1
done

# ---- summary -------------------------------------------------------------
echo ""
echo "=== validate summary ==="
awk -F'\t' '{printf "  %-15s :%-6s %s  %s\n", $1, $2, ($3=="true"?"PASS":"FAIL"), $4}' "$RESULTS"
fail=$(awk -F'\t' '$3=="false"' "$RESULTS" | wc -l | tr -d ' ')
[ "$fail" -gt 0 ] && { echo "$fail validation(s) FAILED"; exit 1; }
echo "all PASS"
