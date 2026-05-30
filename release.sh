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

# ---- 03. clean-tree + branch-sync gate ----------------------------------
# bin/vad gets rebuilt by build_bin.sh; allow it to be modified going in.
step "03 clean-tree + branch-sync gate"
if ! git diff --quiet -- . ':!bin/vad' || ! git diff --cached --quiet -- . ':!bin/vad'; then
    die "working tree dirty (commit / stash; or stale generated client from step 02):
$(git status --short -- . ':!bin/vad')"
fi

# Require: on the release branch (default main) AND fully synced with
# origin. This ties the release to a commit that is publicly visible
# right now — no risk of releasing stale or unpushed code. The contract
# is: merge your release commit via PR like any other change, pull, run
# release.sh.
REL_BRANCH="${RELEASE_BRANCH:-main}"
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$GIT_BRANCH" != "$REL_BRANCH" ]; then
    die "must be on '$REL_BRANCH' to release; you are on '$GIT_BRANCH'.
Set RELEASE_BRANCH=<other> only if you intentionally release from a
non-main branch (e.g. a long-lived release/ branch)."
fi
echo "  fetching origin/$REL_BRANCH to compare …"
if ! git fetch origin "$REL_BRANCH" 2>&1 | sed 's/^/  /'; then
    die "git fetch origin $REL_BRANCH failed (network? auth?)"
fi
LOCAL_SHA=$(git rev-parse HEAD)
REMOTE_SHA=$(git rev-parse "origin/$REL_BRANCH")
BASE_SHA=$(git merge-base HEAD "origin/$REL_BRANCH")
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
    if [ "$LOCAL_SHA" = "$BASE_SHA" ]; then
        die "local '$REL_BRANCH' is BEHIND origin (need: git pull --ff-only)"
    elif [ "$REMOTE_SHA" = "$BASE_SHA" ]; then
        die "local '$REL_BRANCH' is AHEAD of origin (need: git push origin $REL_BRANCH).
The release tag must point at a commit origin already has — don't push
release commits as a side effect of release.sh."
    else
        die "local '$REL_BRANCH' has DIVERGED from origin (reconcile manually)"
    fi
fi
echo "  on $REL_BRANCH @ $(git rev-parse --short HEAD) — fully synced with origin"

GIT_SHA=$(git rev-parse --short HEAD)
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
# Actual file size in human terms. Uses `wc -c` (portable; reports the byte
# count, not the on-disk allocation that `du -h` reports — `du -h` gave us
# inflated numbers like "97M" for an 83 MB file because of filesystem block
# allocation rounding).
human_size() {
    local b
    b=$(wc -c < "$1" | tr -d ' ')
    awk -v b="$b" 'BEGIN {
        split("B KB MB GB TB", units)
        i = 1
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        if (i == 1) printf "%d %s", b, units[i]
        else        printf "%.1f %s", b, units[i]
    }'
}

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
# Project-name-prefixed so the downloaded file is recognisable in a
# Downloads folder that holds bundles from many projects.
ZIP_NAME="accretional-vad-release_${TAG}_${GIT_SHA}.zip"
rm -f "$ZIP_NAME"
zip -qr "$ZIP_NAME" "$LOG_DIR" bin/vad out/
echo "  built: $ZIP_NAME ($(human_size "$ZIP_NAME"))"

# ---- 11. push to GHCR + create GH release -------------------------------
step "11 github_push.sh — push images + create GH release"
TAG="$TAG" ZIP="$ZIP_NAME" bash scripts/github_push.sh 2>&1 | tee "$LOG_DIR/11_github_push.log"

# ---- 12. post-release HTML (download link + provenance) ----------------
step "12 post-release page — opens in browser (non-blocking)"
POST_HTML="$LOG_DIR/post_release.html"
ZIP_SHA=$(sha256_file "$ZIP_NAME")
ACCEPTED_AT=$(awk '/^release ACCEPTED/{print $4, $5, $6}' "$LOG_DIR/release_decision.log" 2>/dev/null)
# Normalize remote URL: strip optional .git suffix, rewrite git@ syntax to
# https, AND strip a trailing slash so concatenating /releases/... doesn't
# produce a double-slash.
REPO_URL=$(git config --get remote.origin.url 2>/dev/null \
    | sed 's,\.git$,,; s,^git@github.com:,https://github.com/,; s,/*$,,')
GH_RELEASE_URL="${REPO_URL}/releases/tag/${TAG}"
ZIP_ABS="$(cd "$(dirname "$ZIP_NAME")" && pwd)/$(basename "$ZIP_NAME")"

# Artifact rows
ARTIFACT_ROWS=""
emit_row() {
    local name="$1" path="$2" size="$3" sha="$4"
    ARTIFACT_ROWS="${ARTIFACT_ROWS}<tr><td>${name}</td><td><code>${path}</code></td><td>${size}</td><td><code>${sha}</code></td></tr>"
}
emit_row "Release zip" "$ZIP_ABS" "$(human_size "$ZIP_NAME")" "$ZIP_SHA"
[ -f bin/vad ] && emit_row "Native binary" "bin/vad" "$(human_size bin/vad)" "$(sha256_file bin/vad)"
for arch in amd64 arm64; do
    f="out/${arch}/vad"
    [ -f "$f" ] && emit_row "Fat linux/${arch} binary" "$f" "$(human_size "$f")" "$(sha256_file "$f")"
done
for arch in amd64 arm64; do
    img="vad:${arch}"
    if id=$(docker image inspect -f '{{.Id}}' "$img" 2>/dev/null); then
        emit_row "Container image (${arch})" "$img" "$(docker images --format '{{.Size}}' "$img" | head -1)" "${id##sha256:}"
    fi
done

cat > "$POST_HTML" <<EOF
<!DOCTYPE html><html><head><meta charset="utf-8"><title>${TAG} released</title>
<style>
body{font:14px/1.5 system-ui,sans-serif;max-width:920px;margin:32px auto;padding:0 16px;color:#1d1d1f}
h1{font-size:24px;margin:0 0 4px}.subtitle{color:#666;margin-bottom:22px}
section{background:#f7f7f9;border-radius:8px;padding:14px 18px;margin:12px 0}
section h2{margin:0 0 8px;font-size:13px;color:#444;text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:6px 10px 6px 0;text-align:left;vertical-align:top;border-bottom:1px solid #eee}
th{color:#555;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
code{font-family:ui-monospace,Menlo,monospace;font-size:12px;word-break:break-all}
a{color:#06c;text-decoration:none}a:hover{text-decoration:underline}
.dl{display:inline-block;background:#1b7e1b;color:white;padding:10px 22px;border-radius:6px;font-weight:600;margin-top:8px}
.dl:hover{background:#1d8a1d;text-decoration:none}
</style></head><body>
<h1>✅ ${TAG} released</h1>
<div class="subtitle">Pushed to <a href="${REPO_URL}">${REPO_URL}</a>${REPO_URL:+ — }<a href="${GH_RELEASE_URL}">view on GitHub</a></div>

<section>
  <h2>Download</h2>
  <a class="dl" href="file://${ZIP_ABS}" download>Download ${ZIP_NAME} ($(human_size "$ZIP_NAME"))</a>
  <p style="margin-top:10px;color:#555">sha256: <code>${ZIP_SHA}</code></p>
</section>

<section>
  <h2>Release</h2>
  <table>
    <tr><th>Tag</th><td><code>${TAG}</code></td></tr>
    <tr><th>Commit</th><td><code>${GIT_SHA}</code> on ${GIT_BRANCH}</td></tr>
    <tr><th>Repo</th><td>${REPO_URL}</td></tr>
    <tr><th>Accepted at</th><td>${ACCEPTED_AT}</td></tr>
    <tr><th>Accepted by</th><td>${USER:-unknown}@$(hostname -s)</td></tr>
    <tr><th>Build host</th><td>$(uname -s)/$(uname -m)</td></tr>
  </table>
</section>

<section>
  <h2>Artifacts &amp; checksums</h2>
  <table>
    <tr><th>Name</th><th>Path / Ref</th><th>Size</th><th>sha256 / image id</th></tr>
    ${ARTIFACT_ROWS}
  </table>
</section>

<section>
  <h2>Pull the container</h2>
  <pre><code>docker pull ${REGISTRY:-ghcr.io/accretional/vad}:${TAG}
docker run --rm -p 50051:50051 ${REGISTRY:-ghcr.io/accretional/vad}:${TAG}</code></pre>
</section>
</body></html>
EOF

echo "  wrote: $POST_HTML"
if command -v open >/dev/null 2>&1; then open "$POST_HTML"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$POST_HTML"
fi

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
