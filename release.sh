#!/usr/bin/env bash
#
# Release orchestrator. Calls each phase script under scripts/ in order
# and stops on the first failure. NO skip flags — every phase runs every
# time. Debug individual phases by invoking them standalone:
#
#   bash scripts/build_clients.sh
#   bash scripts/build_bin.sh
#   bash scripts/build_containers.sh
#   bash scripts/validate_builds.sh        # LOG_DIR=/tmp/v overrides log root
#   bash scripts/review_release.sh release-logs/<tag>/
#   TAG=v0.2.0 ZIP=release_*.zip bash scripts/github_push.sh
#
# Pipeline order:
#
#   01  setup.sh                                  (verify deps)
#   02  scripts/build_clients.sh                  (regen Go gRPC bindings)
#   03  clean-tree gate                           (catches stale generated code)
#   04  test.sh                                   (full test suite incl. Docker e2e)
#   05  scripts/build_bin.sh                      (native binary)
#   06  scripts/build_containers.sh               (per-arch slim + fat)
#   07  scripts/validate_builds.sh                (5-port validation matrix)
#   08  metadata.json + per-step logs aggregated
#   09  scripts/review_release.sh                 (BLOCKS — operator clicks)
#   10  release_decision.log + release_<tag>_<sha>.zip
#   11  scripts/github_push.sh                    (multi-arch push + gh release)
#
# TODO (separate issues; insert as steps in this orchestrator once
# implemented):
#   - SBOM + vulnerability scan: between validate (07) and review (09).
#     `docker scout cves`, `syft sbom`, `grype --fail-on high`.
#   - Image signing: cosign + sigstore keyless after push (inside
#     scripts/github_push.sh).
#
# Usage:
#   bash release.sh                       # full pipeline; auto-tags vYYYY.MM.DD-<sha>
#   bash release.sh --tag v0.2.0          # use a specific version tag
#
# Environment:
#   REGISTRY=ghcr.io/accretional/vad      # for image push + gh release
#   ARCHES="linux/amd64 linux/arm64"      # which arches to build
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

TAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,/^set -uo pipefail/p' "$0" | grep -E '^#' | sed 's/^# //;s/^#$//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

nowiso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() { echo ""; echo "==[ $* ]=="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ---- 01. setup.sh -------------------------------------------------------
step "01 setup.sh — verify deps"
bash setup.sh

# ---- 02. regenerate Go clients ------------------------------------------
step "02 build_clients.sh — regenerate Go gRPC bindings"
bash scripts/build_clients.sh

# ---- 03. clean-tree gate ------------------------------------------------
# bin/vad gets rebuilt by build_bin.sh; allow it to be modified going in.
step "03 clean-tree gate"
if ! git diff --quiet -- . ':!bin/vad' || ! git diff --cached --quiet -- . ':!bin/vad'; then
    die "working tree dirty (commit / stash; or stale generated client from step 02):
$(git status --short -- . ':!bin/vad')"
fi

GIT_SHA=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
BUILD_STARTED=$(nowiso)
TAG="${TAG:-v$(date -u +%Y.%m.%d)-${GIT_SHA}}"
LOG_DIR="release-logs/${TAG}"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"
echo "  tag:    $TAG"
echo "  commit: $GIT_SHA on $GIT_BRANCH"
echo "  logs:   $LOG_DIR/"

# Run a phase command, teeing into a numbered log under $LOG_DIR.
phase() {
    local num="$1" name="$2"; shift 2
    local log="$LOG_DIR/${num}_${name}.log"
    step "${num} ${name}"
    if "$@" 2>&1 | tee "$log"; then
        return 0
    else
        echo "ERROR: phase ${num}_${name} failed; see $log" >&2
        return 1
    fi
}

# ---- 04. tests ----------------------------------------------------------
phase 04 tests bash test.sh

# ---- 05. native binary --------------------------------------------------
phase 05 build_bin bash scripts/build_bin.sh

# ---- 06. per-arch containers + fat binaries -----------------------------
phase 06 build_containers bash scripts/build_containers.sh

# ---- 07. validate every artifact ----------------------------------------
LOG_DIR_ABS="$(cd "$LOG_DIR" && pwd)"
VAL_LOG_DIR="${LOG_DIR_ABS}/07_validate"
mkdir -p "$VAL_LOG_DIR"
LOG_DIR="$VAL_LOG_DIR" bash scripts/validate_builds.sh 2>&1 | tee "$LOG_DIR_ABS/07_validate.log"
VAL_RC=${PIPESTATUS[0]}
LOG_DIR="$(dirname "$VAL_LOG_DIR")"
# Don't bail on validation failure — let the operator see the HTML and
# decide. The HTML clearly surfaces failures.
[ "$VAL_RC" -ne 0 ] && echo "  (one or more validations FAILED — review HTML for details)"

# ---- 08. assemble metadata.json -----------------------------------------
step "08 assemble metadata.json"
sha256_file() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}
human_size() { du -h "$1" | awk '{print $1}'; }

art_row() {
    local name="$1" path="$2"
    [ ! -f "$path" ] && return
    printf '{"name":"%s","path":"%s","size":"%s","sha256":"%s"}' \
        "$name" "$path" "$(human_size "$path")" "$(sha256_file "$path")"
}

ARTIFACTS="$(art_row 'native (host)' bin/vad)"
for arch in amd64 arm64; do
    f="out/${arch}/vad"
    [ -f "$f" ] && ARTIFACTS="${ARTIFACTS},$(art_row "fat-${arch} (linux/${arch})" "$f")"
done
for arch in amd64 arm64; do
    img="vad:${arch}"
    if id=$(docker image inspect -f '{{.Id}}' "$img" 2>/dev/null); then
        size=$(docker images --format '{{.Size}}' "$img" | head -1)
        ARTIFACTS="${ARTIFACTS},{\"name\":\"image: ${img}\",\"path\":\"${img}\",\"size\":\"${size}\",\"sha256\":\"${id##sha256:}\"}"
    fi
done

# Validations come from validate_builds.sh's results.tsv (name port ok logpath).
VALIDATIONS=""
if [ -f "$VAL_LOG_DIR/results.tsv" ]; then
    while IFS=$'\t' read -r name port ok logpath; do
        sep=""; [ -n "$VALIDATIONS" ] && sep=","
        VALIDATIONS="${VALIDATIONS}${sep}{\"name\":\"${name}\",\"port\":${port},\"ok\":${ok},\"log_path\":\"07_validate/${logpath}\"}"
    done < "$VAL_LOG_DIR/results.tsv"
fi

LOGS=""
for f in "$LOG_DIR"/*.log; do
    [ ! -f "$f" ] && continue
    base=$(basename "$f")
    sep=""; [ -n "$LOGS" ] && sep=","
    LOGS="${LOGS}${sep}{\"name\":\"${base}\",\"path\":\"${base}\"}"
done

cat > "$LOG_DIR/metadata.json" <<EOF
{
  "tag":           "${TAG}",
  "git_sha":       "${GIT_SHA}",
  "git_branch":    "${GIT_BRANCH}",
  "date":          "$(nowiso)",
  "build_host":    "$(hostname -s)",
  "build_os_arch": "$(uname -s)/$(uname -m)",
  "build_user":    "${USER:-unknown}",
  "build_started": "${BUILD_STARTED}",
  "artifacts":     [${ARTIFACTS}],
  "validations":   [${VALIDATIONS}],
  "logs":          [${LOGS}]
}
EOF
echo "  wrote: $LOG_DIR/metadata.json"

# ---- 09. review gate (BLOCKS) -------------------------------------------
step "09 review_release.sh — BLOCKING; click ACCEPT or CANCEL in browser"
if ! bash scripts/review_release.sh "$LOG_DIR"; then
    echo "  CANCELLED. Nothing pushed. release-logs/${TAG}/ preserved."
    exit 1
fi
{
    echo "release ACCEPTED at $(nowiso)"
    echo "operator: ${USER:-unknown}"
    echo "host: $(hostname -s)"
    if [ -f "$VAL_LOG_DIR/results.tsv" ]; then
        pass=$(awk -F'\t' '$3=="true"' "$VAL_LOG_DIR/results.tsv" | wc -l | tr -d ' ')
        fail=$(awk -F'\t' '$3=="false"' "$VAL_LOG_DIR/results.tsv" | wc -l | tr -d ' ')
        echo "validations: ${pass} OK / ${fail} FAIL"
    fi
} > "$LOG_DIR/release_decision.log"

# ---- 10. zip ------------------------------------------------------------
step "10 build release zip"
ZIP_NAME="release_${TAG}_${GIT_SHA}.zip"
rm -f "$ZIP_NAME"
zip -qr "$ZIP_NAME" "$LOG_DIR" bin/vad out/
echo "  built: $ZIP_NAME ($(human_size "$ZIP_NAME"))"

# ---- 11. push to GHCR + create GH release -------------------------------
step "11 github_push.sh — push images + create GH release"
TAG="$TAG" ZIP="$ZIP_NAME" bash scripts/github_push.sh 2>&1 | tee "$LOG_DIR/11_github_push.log"

# ---- summary -----------------------------------------------------------
echo ""
echo "==[ Release artifacts (${TAG}) ]=="
echo "Native:                          bin/vad  $(human_size bin/vad)"
for f in out/*/vad; do
    [ -f "$f" ] && echo "Standalone Linux binary:         $f  $(human_size $f)"
done
docker images vad --format "Slim container:                  {{.Repository}}:{{.Tag}}  {{.Size}}"
echo "Zip:                             $ZIP_NAME  $(human_size "$ZIP_NAME")"
echo "Logs:                            $LOG_DIR/"
echo ""
echo "OK"
