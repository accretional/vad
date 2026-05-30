# Release pipeline

How `bash release.sh` turns a clean commit into a tagged GitHub release
with multi-arch images, standalone binaries, and a permanent build
record. This doc covers the orchestrator, the per-phase scripts, the
review gate, and what's planned next.

## Design rules

- **No skip flags.** Every release runs every phase. If you want to
  iterate on a single phase (e.g. the validator), run that one script
  directly — don't add a `--skip-X` to the orchestrator.
- **One phase, one script.** Each phase lives at `scripts/<phase>.sh`
  and is self-contained: it can be invoked standalone for debugging.
- **Every artifact is validated** end-to-end against a real gRPC server
  before the review gate sees it.
- **The release gate is human.** A browser-based ACCEPT/CANCEL click
  is the last thing standing between you and a push.

## Phase order

```
01  setup.sh                        verify deps
02  scripts/build_clients.sh        regen Go (+ optional Python) gRPC bindings
03  clean-tree + branch-sync gate   on RELEASE_BRANCH (default main), HEAD == origin
04  test.sh                         go vet, unit, integration, Docker e2e + fetch
05  scripts/build_bin.sh            native binary (./bin/vad for the host)
06  scripts/build_containers.sh     per-arch slim containers + fat host binaries
07  scripts/validate_builds.sh      5-port RPC validation matrix
08  aggregate metadata.json + logs
09  scripts/review_release.sh       BLOCKING — operator clicks ACCEPT or CANCEL
10  release_decision.log + accretional-vad-release_<tag>_<sha>.zip
11  scripts/github_push.sh          multi-arch push + git tag + gh release
12  post_release.html               opens in browser; download link + provenance
```

Failure in any phase aborts the pipeline. Review CANCEL exits non-zero
with nothing pushed; release-logs/`<tag>`/ is preserved for postmortem.

## What touches git / GitHub

The pipeline is deliberately conservative about upstream state. It
**reads** git in many phases (to know the tag, branch, SHA) but only
**writes** to GitHub at the very end, and never pushes your branch.

| Phase | Reads | Writes |
|---|---|---|
| 03 gate | branch name, HEAD, `origin/<RELEASE_BRANCH>` (after `git fetch`) | nothing |
| 04–10  | HEAD, branch | nothing |
| 11 push | HEAD | `docker buildx --push` → GHCR; `git tag -a <TAG>` + `git push origin <TAG>`; `gh release create <TAG> ... <zip> <binaries>` |
| 12 page | accept-log | local HTML only |

In particular the pipeline does **not** push your release branch as a
side effect. The contract is: *the commit being released is exactly
what's on `origin/<RELEASE_BRANCH>` right now*. You merge your release
commit via PR like any other change, `git pull`, then `bash release.sh`.

The phase-03 gate fails fast (before tests, before building) on any of:
- not on `$RELEASE_BRANCH` (default `main`; override via env var)
- local HEAD behind origin (`git pull --ff-only` first)
- local HEAD ahead of origin (`git push` first; you don't want
  release.sh sneaking your unreviewed commits to main)
- diverged (reconcile manually)

What lands on GitHub at phase 11:
- **Container manifest** at `${REGISTRY:-ghcr.io/accretional/vad}:<TAG>`
  (plus a `:latest` alias) — pullable with
  `docker pull ghcr.io/accretional/vad:<TAG>`. Part of GitHub Packages.
- **A git tag** at the commit you released from. The tag is annotated
  with `Release <TAG>` and pushed to origin.
- **A GitHub Release** at `https://github.com/<org>/vad/releases/tag/<TAG>`
  with release notes auto-generated from PR titles since the previous
  tag (`gh release create --generate-notes`). The release page carries
  the **release zip + standalone binaries** as downloadable assets.

## The per-phase scripts

Each lives at `scripts/<name>.sh`. Run standalone for fast iteration.

### `scripts/build_clients.sh`

Regenerates the Go gRPC client + server bindings from `proto/vad.proto`
into `proto/vadpb/`. Also generates the Python client into `proto/vadpy/`
when `grpcio-tools` is installed (`pip install grpcio-tools`). C++ and
Java are TODOs — each is one extra `--<lang>_out` plugin invocation.

Idempotent — protoc writes the same bytes given the same input + plugin
versions. Running this on a clean tree leaves no diff. Used at the top
of `release.sh` so stale generated code fails the clean-tree gate before
anything else builds.

### `scripts/build_bin.sh`

Thin wrapper around `build-native.sh`. Produces `./bin/vad`, the
self-contained native binary for the host OS/arch (`darwin/arm64` on an
M-series Mac; `linux/amd64` on an x86 Linux build host). ~59 MB.

### `scripts/build_containers.sh`

Two artifacts per architecture via two `docker buildx` invocations
against the same Dockerfile (different `--target`):

| Target | Result | Output |
|---|---|---|
| `--target runtime --load` | slim container image | `vad:amd64`, `vad:arm64` in local docker daemon |
| `--target export --output type=local,dest=out/<arch>` | fat standalone Linux binary | `out/amd64/vad`, `out/arm64/vad` |

Requires buildx (`brew install docker-buildx`) and a docker-container
builder (auto-created on first run as `vad-builder`).

Cross-arch emulation needs `colima start --vz-rosetta` on Apple Silicon
(qemu user-mode emulation crashes Go's HTTP/2 client during `go mod
download`; Rosetta-backed colima works reliably). On x86 Linux there's
no emulation cost in either direction.

### `scripts/validate_builds.sh`

Stands up each of the 5 built artifacts on its own port and runs the
Go-based `tests/validate` against it (Detect on silence + Fetch for
every backend in the proto enum). Results land in `$LOG_DIR/results.tsv`
and per-artifact `.log` files. Returns non-zero if any artifact failed.

Port allocation (deliberately high to dodge dev defaults):

| Artifact | Port | How it's started |
|---|---|---|
| native `bin/vad` | 50100 | local exec |
| fat `linux/amd64` binary | 50101 | wrapped in `debian:bookworm-slim`, `docker run` |
| fat `linux/arm64` binary | 50102 | same |
| slim `vad:amd64` container | 50103 | `docker run --platform linux/amd64` |
| slim `vad:arm64` container | 50104 | `docker run --platform linux/arm64` |

### `scripts/review_release.sh`

Wraps `cmd/release-review`, a tiny embedded-HTTP server that:

1. Reads `release-logs/<tag>/metadata.json`
2. Renders an HTML review page (provenance, artifact `sha256` + size,
   per-artifact PASS/FAIL with deep-links to the validation logs)
3. `open`s it in the operator's default browser
4. Blocks until the operator clicks **ACCEPT & RELEASE** or **CANCEL**

The server binary is built into the log dir itself (`_release-review`)
so it gets zipped along with the build record. Exit 0 on accept; exit 1
on cancel (which `release.sh` propagates).

### `scripts/github_push.sh`

Multi-arch push to `${REGISTRY:-ghcr.io/accretional/vad}` via buildx,
then `git tag -a <tag>`, then `gh release create <tag>
--generate-notes` with the zip + raw binaries uploaded.

Release notes come from `--generate-notes` — GitHub auto-generates them
from PR titles since the last tag. No separate `CHANGELOG.md` needed.

`RELEASE_DRY_RUN=1` env var prints the docker / git / gh commands the
script would run, without executing any of them. Lets you exercise
`release.sh` end-to-end locally without creating real git tags or
touching the registry. This is a *mode*, not a *skip* — every phase
still runs, and the dry run prints exactly what the real push would do.

## The review gate

When phase 09 fires, `release-review` reads `metadata.json` and renders
an HTML page in the operator's browser. The page has four sections:

1. **Provenance** — tag, git SHA, branch, build host, OS/arch, operator
2. **Artifacts** — name, path, `sha256`, size for every built thing
   (zip, native binary, fat per-arch binaries, container image IDs)
3. **Validation results** — PASS/FAIL per artifact with a `<a>` linking
   each to the per-artifact validation log served at `/logs/`
4. **All logs** — every `.log` file under `release-logs/<tag>/`,
   directly linked for inspection

Below: two big buttons — **ACCEPT & RELEASE** (green) and **CANCEL**
(grey). The page POSTs `/decision` with the verdict; the server shuts
down and exits 0 / 1 accordingly.

Once `release.sh` reaches phase 12, a separate `post_release.html`
opens in the browser with the download link, zip `sha256`, accepted-at
timestamp, accepted-by user, and a "pull the container" snippet —
permanent record under `release-logs/<tag>/post_release.html`.

### Planned: hosted review + MFA

The current review gate runs on the operator's localhost. The roadmap
is a hosted review service:

- Operator triggers a release via `release.sh` (or CI on push to a
  release branch). The same metadata + logs are uploaded to a hosted
  review URL.
- The reviewer (operator OR a designated approver — possibly a
  different person from the one who started the build) gets a
  notification with the URL.
- Clicking ACCEPT requires a fresh **MFA challenge** — a hardware-key
  tap, OIDC step-up, or push notification to a registered device —
  rather than a single bare browser click.
- The signed approval is captured alongside the release record (who
  approved, when, with what factor), forming an auditable provenance
  chain that survives the local `release-logs/` dir.

This unlocks several deliberate properties: two-person release control
(builder ≠ approver), tamper-evident approval, and a single canonical
record visible across the org rather than scattered across operator
laptops.

Not implemented yet. The current localhost flow is the placeholder.

## Outputs

```
release-logs/<tag>/
  01_setup.log                  Per-phase tee'd logs.
  02_build_clients.log
  04_tests.log
  05_build_bin.log
  06_build_containers.log
  07_validate.log               Top-level validate output.
  07_validate/
    native.log                  Per-artifact validation detail.
    fat-amd64.log
    fat-arm64.log
    slim-amd64.log
    slim-arm64.log
    results.tsv                 Machine-readable PASS/FAIL table.
    _validate                   The validator binary (zipped along).
  11_github_push.log
  metadata.json                 The single-source-of-truth file the
                                review HTML + release zip both read.
  _release-review               The review server binary (zipped along).
  release_decision.log          Operator's verdict + timestamp.
  post_release.html             Final download page.

release_<tag>_<sha>.zip         Bundle uploaded to the GitHub release.
                                Contains release-logs/<tag>/ + bin/vad
                                + out/{amd64,arm64}/vad.
```

## TODO (separate work items)

These are documented hooks in `release.sh` and `scripts/github_push.sh`
that haven't been implemented yet:

- **SBOM + vulnerability scan** between validate (07) and review (09):
  `docker scout cves`, `syft sbom`, `grype --fail-on high`. Counts
  would surface in the metadata + HTML.
- **Image signing** (cosign + sigstore keyless) after each push inside
  `github_push.sh`. Sign each per-arch image and the multi-arch
  manifest; cache signing receipts under `release-logs/<tag>/`.
- **Hosted review + MFA approval** (see *Planned* section above).
- **Additional language clients** (Python is wired; C++ + Java are
  one-line additions to `build_clients.sh` each).

## Debugging individual phases

```bash
# Regen clients, see the diff (or lack thereof):
bash scripts/build_clients.sh && git diff proto/vadpb/

# Native binary only:
bash scripts/build_bin.sh

# Container builds only, just amd64:
ARCHES=linux/amd64 bash scripts/build_containers.sh

# Validation matrix against whatever's currently built:
LOG_DIR=/tmp/vad-val bash scripts/validate_builds.sh

# Re-open the review page for an old release record:
bash scripts/review_release.sh release-logs/<old-tag>/

# Dry-run a push end-to-end:
RELEASE_DRY_RUN=1 TAG=v0.0.0 ZIP=release_test.zip \
  bash scripts/github_push.sh
```

## Quick reference

```bash
bash release.sh                       # full pipeline, auto-tags vYYYY.MM.DD-<sha>
bash release.sh --tag v0.2.0          # use a specific version tag
RELEASE_DRY_RUN=1 bash release.sh     # full pipeline; push step is logged-only
```
