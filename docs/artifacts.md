# Artifacts: binaries and containers

A single `release.sh` invocation produces three classes of shippable
artifact: one **native binary**, two **standalone Linux binaries**
(one per arch), and two **slim container images** (one per arch). This
doc covers what each is, how it's built, and when to reach for it.

## At a glance

| Artifact | Purpose | Embeds | Pulls from disk | Typical size |
|---|---|---|---|---|
| `bin/vad` | Native dev binary for the build host | ORT dylib + all 5 backends | only fallback | ~59 MB |
| `out/amd64/vad` | Standalone `linux/amd64` executable | ORT `.so` + all 5 backends | only fallback | ~43 MB |
| `out/arm64/vad` | Standalone `linux/arm64` executable | ORT `.so` + all 5 backends | only fallback | ~39 MB |
| `vad:amd64` image | Production container for `linux/amd64` | ORT + default backend only | `/onnx/{ort,weights}/` | ~235 MB |
| `vad:arm64` image | Production container for `linux/arm64` | ORT + default backend only | `/onnx/{ort,weights}/` | ~248 MB |

Plus, after `release.sh`, the **release zip** (`release_<tag>_<sha>.zip`,
~96 MB) — bundles all of the above (minus the container images, which
live in the registry once pushed) along with the full build log
record under `release-logs/<tag>/`.

## Build modes: fat vs slim

The same Go source tree builds two binary variants via a build tag:

```bash
go build              ./cmd/vad   # "fat":  embeds ORT + all 5 weights
go build -tags slim   ./cmd/vad   # "slim": embeds ORT + default backend only
```

Two file pairs implement the divergence:

- `internal/embedded/weights_fat.go` (`//go:build !slim`):
  `//go:embed all:weights` — every backend's `.onnx` baked in.
- `internal/embedded/weights_slim.go` (`//go:build slim`):
  `//go:embed all:weights/pyannote` — only the default model embedded.
- `cmd/vad/disk_fat.go`: `onDiskWeightsRoot = "weights"` — relative to cwd.
- `cmd/vad/disk_slim.go`: `onDiskWeightsRoot = "/onnx/weights"` — the
  container convention.

This is the right pattern for further per-mode divergence later: any
flag, default, or behavior that needs to differ between fat and slim
adds another build-tagged file pair.

## Resolution: disk-first everywhere

Both variants use the same resolution order in
`internal/embedded/embedded.go`:

1. **Disk** at `<onDiskWeightsRoot>/<backend>/` — used in the slim
   container (Dockerfile COPYs the full weights tree there) and in dev
   (changes to `weights/<backend>/` take effect without rebuilding).
2. **Embed** — fallback for any backend not on disk. In slim builds
   only `pyannote` is embedded, so requesting any other backend on a
   host without `/onnx/weights/<backend>/` fails with a clear error.

Same logic for the ORT dylib (`discoverOnnxRuntime` in `cmd/vad/main.go`
prepends `/onnx/ort/<libname>` to its candidate list; falls through to
system paths; embed materialization is the deep fallback).

This means the slim container never `mkdirtemp`s — both the ORT dylib
and every model load directly from `/onnx/`. The embed lives in the
binary as a safety net for when the slim binary gets extracted from the
container and run standalone (e.g. for ops debugging).

## The container Dockerfile

Three stages:

```
builder    golang:1.26.1-bookworm
           Downloads ORT, runs prep-embed.sh, builds BOTH:
              /app/vad-fat    (default build; embeds everything)
              /app/vad-slim   (-tags slim; embeds ORT + default model)

export     scratch
           Single file:  /vad = the fat binary
           Pulled out via `docker buildx build --target export
           --output type=local,dest=out/<arch>` — this is how the
           release pipeline harvests the standalone host binaries.

runtime    debian:bookworm-slim
           /server               the slim binary
           /onnx/weights/<5>/    full weights tree COPY'd from builder
           /onnx/ort/            the ORT shared lib COPY'd from builder
           ENTRYPOINT ["/server"]
           EXPOSE 50051
```

`docker build` defaults to the last stage (`runtime`) so a regular
`docker build -t vad .` produces the slim container.

## Running each artifact

### Native binary

```bash
./bin/vad                        # default: pyannote on :50051
./bin/vad -backend silero
./bin/vad -port 50055 -backend marblenet
```

### Standalone Linux binaries

```bash
# Copy out/amd64/vad to a Linux x86_64 host and run:
./vad
./vad -backend silero

# Or wrap it in a minimal debian-slim image yourself:
FROM debian:bookworm-slim
COPY vad /vad
ENTRYPOINT ["/vad"]
```

### Slim container

```bash
docker run --rm -p 50051:50051 vad:amd64
docker run --rm -p 50051:50051 vad:amd64 -backend silero

# From the published manifest (when --release is used):
docker pull ghcr.io/accretional/vad:latest
docker run --rm -p 50051:50051 ghcr.io/accretional/vad:latest
```

The slim container expects `/onnx/{ort,weights}/` to exist. The shipped
image populates them; if you want to swap weights for a custom build
you can bind-mount your own:

```bash
docker run --rm -p 50051:50051 \
  -v /my/weights:/onnx/weights:ro \
  ghcr.io/accretional/vad:latest -backend silero
```

The on-disk weights win over the embed in slim mode, so you can replace
just one backend's `model.onnx` without touching the others.

## When to use which

| You want to … | Use |
|---|---|
| Run on your dev Mac while editing the codebase | `bash build-native.sh && ./bin/vad` |
| Drop a single binary on a bare Linux box | `out/<arch>/vad` from a release |
| Deploy to k8s / Cloud Run / Docker host | `ghcr.io/accretional/vad:<tag>` (slim image) |
| Swap weights without rebuilding | slim image + `-v /my/weights:/onnx/weights:ro` |
| Run a quick test in a one-off container | `vad:<arch>` (locally built) |
| Distribute a self-contained "no Docker needed" binary | fat standalone Linux binary |

## Provenance + checksums

Every release's `post_release.html` lists every artifact with `sha256`
(image ID for container images). The same data lives in
`release-logs/<tag>/metadata.json` as machine-readable JSON, which
both the review HTML and the release zip pull from.

The release zip (`release_<tag>_<sha>.zip`) is the canonical bundle —
it includes the full build log record, validation results, the
post-release page, and the binaries themselves. Container images live
in the registry; pull them by tag.

See [release-pipeline.md](release-pipeline.md) for how all of this is
built and validated.
