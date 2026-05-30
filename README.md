# vad

High-throughput gRPC + go + ONNX Voice Activity Detection with CPU model serving

* **>200× realtime VAD throughput with CPU inference**
* serves 5 SoTA VAD models via gRPC, container, go package, or self-contained binary file
* High Throughput Go Voice AI inference stack, 4 ports PyTorth->ONNX, 1 port NeMo->ONNX
* Realtime in-browser VAD through transformers.js / onnxruntime-web, can act as an ONNX model discovery service and model host to webGPU applications
* Websocket interface for realtime remote VAD computed on CPUs; single-binary server or prebuilt container you can use to launch instantly

<img width="1014" height="956" alt="Screenshot 2026-05-30 at 1 33 57 PM" src="https://github.com/user-attachments/assets/c89528f7-c038-4107-ae5c-4b6852362909" />

# Overview

The vad gRPC server binary (`bin/vad`, ~59 MB on darwin/arm64) bundles all 5 backends' ONNX weights and the ONNX Runtime dylib via `go:embed` into **a single self-contained binary**. Bidirectional streaming over WebSocket and gRPC supports **>100× realtime audio VAD processing per CPU core**, with the multi-backend comparison demo at [`cmd/basic-vad-web/`](cmd/basic-vad-web/) (`./cmd/basic-vad-web/run.sh` to launch locally).

Models: **pyannote**, **fsmn**, **firered**, **silero**, and **marblenet** VAD all run from a unified ONNX path, with Kaldi- and NeMo-equivalent feature extractors in pure Go (`fbank/`, `melspec/`), parity-tested against the upstream PyTorch / NeMo references to within float32 round-off.


All backends speak the same `vad.VoiceSegmentation` proto (`Detect` unary + `DetectStream` bidi streaming + `Fetch` for weights/URL handoff), so clients are backend-agnostic. As a gRPC service it natively supports client libraries in C++, Python, Go, Java, and other major language runtimes.

Build from source with `bash build-native.sh` — needs Go (and ffmpeg for the demo's audio decoding side-path). Or build the container with `docker build -t vad .` and run `docker run -p 50051:50051 vad`. MIT License.

## Models

All five backends ship inside `bin/vad` via `go:embed`. Bench numbers come from the parity reports under [`speax/benchmarks/out/<backend>-onnx/`](https://github.com/accretional/speax/tree/main/benchmarks/out) (ONNX Runtime 1.26.0, CPU EP, warmed, 10 s reference clip).

| `-backend` | Model | Team | Params | Size (ONNX) | Original format | Latency (10s, CPU) | Speaker IDs |
|---|---|---|---|---|---|---|---|
| `pyannote` (default) | Pyannote Segmentation 3.0 | pyannote · [[gh]](https://github.com/pyannote/pyannote-audio) [[hf]](https://huggingface.co/pyannote/segmentation-3.0) [[site]](https://www.pyannote.ai/) | ~1.47 M | ~6 MB | PyTorch | ~50 ms | yes (up to 3) |
| `fsmn` | FunASR FSMN-VAD | Alibaba DAMO · [[gh]](https://github.com/modelscope/FunASR) [[hf]](https://huggingface.co/funasr/fsmn-vad) [[site]](https://www.funasr.com/) | ~440 K | 1.73 MB | PyTorch | 2.4 ms | no |
| `firered` | FireRed DFSMN-VAD | FireRedTeam (RedNote) · [[gh]](https://github.com/FireRedTeam/FireRedASR) [[hf]](https://huggingface.co/FireRedTeam/FireRedVAD) [[site]](https://fireredteam.github.io/) | ~600 K | 2.34 MB | PyTorch | 4.67 ms | no |
| `silero` | Silero VAD | Silero · [[gh]](https://github.com/snakers4/silero-vad) [[hf]](https://huggingface.co/onnx-community/silero-vad) [[site]](https://silero.ai/) | ~129 K | 1.26 MB | PyTorch JIT | 66.5 µs/chunk (~21 ms / 10 s) | no |
| `marblenet` | NVIDIA Frame VAD Multilingual MarbleNet v2.0 | NVIDIA NeMo · [[gh]](https://github.com/NVIDIA/NeMo) [[hf]](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) [[site]](https://developer.nvidia.com/nemo-framework) | 91,378 | 362 KB | NeMo (.nemo, PyTorch under the hood) | 0.62 ms | no |

Pure-Go log-Mel features live in two parity-tested packages:
- `fbank/` — Kaldi-style (Povey/Hamming window, HTK mel, snip_edges); used by FSMN + FireRed. Parity-tested against `kaldi_native_fbank`. See [`fbank/README.md`](fbank/README.md).
- `melspec/` — NeMo-style (Hann window, Slaney mel, center reflection padding, `log(mel + 2⁻²⁴)`); used by MarbleNet. Parity-tested against NeMo's `AudioToMelSpectrogramPreprocessor`.

## Platforms

What works where today. Bundled `bin/vad` is built per host triple (no fat binary yet); cross-compile by adjusting `GOOS`/`GOARCH` and the embedded ORT dylib in `internal/embedded/`.

### macOS (arm64)

| Target | Status | Notes |
|---|---|---|
| CPU | DONE | Default ORT CPU EP. All 5 backends run end-to-end. |
| GPU (Metal) | SOON | ORT 1.22+ ships CoreML EP for darwin/arm64; not wired into `NewDynamicAdvancedSession` calls yet (see `pkg/vad/*.go`). |
| NPU (Apple Neural Engine) | SOON | Same CoreML EP — opt-in flag would route eligible ops to the ANE. |
| Browser (transformers.js / onnxruntime-web) | DONE | Demo runs all 5 models WASM-side via Web Workers; `Fetch` RPC hands out CDN URLs from per-backend `url.txt` sidecars. |
| Go package | DONE | `import "github.com/accretional/vad/pkg/vad"` — see `pkg/vad/`. |
| gRPC service | DONE | `./bin/vad` exposes `vad.VoiceSegmentation` (`Detect`, `DetectStream`, `Fetch`). |
| Standalone app (demo) | DONE | `./cmd/basic-vad-web/run.sh` brings up vad + audio + demo HTTP. |
| Container (Docker / colima / OrbStack) | DONE | Self-contained `linux/amd64` image (~70 MB) built via `docker build -t vad .` — ORT dylib + all 5 backends' weights embedded via `go:embed`, no `/weights` mount or `ONNXRUNTIME_LIB` needed. `docker run -p 50051:50051 vad` works out of the box; per-backend e2e matrix in `tests/e2e/all_backends_test.go`. |

### Linux (x86_64)

| Target | Status | Notes |
|---|---|---|
| CPU | DONE (in container) / PENDING (native) | The Dockerfile produces a fully self-contained `linux/amd64` binary today (`internal/embedded/ort_linux_amd64.go` wires the embed). Native (non-container) Linux build needs validation on actual Linux hardware before flipping the column. |
| GPU (CUDA / TensorRT) | SOON | — |
| Browser | PENDING | Same WASM bundle; needs Linux-host validation. |
| Go package | PENDING | — |
| gRPC service | PENDING | — |

### Windows
| Target | Status | Notes |
|---|---|---|
| Downloadable Executable | MAYBE | Reach out if you're interested in testing/validating support across Windows modalities! |

### "Beyond the Edge", Ultra-low power/ambient inference

| Target | Status | Notes |
|---|---|---|
| Edge | SOON | Low latency edge inference |
| "The Board" | CONTACT US | Voice AI inference, in any shape |
| "The Dongle" | CONTACT US | Voice AI inference, over the Serial Bus |
| "The Wire" | CONTACT US | Cutting edge Network-level Voice inference hardware |


## Interfaces

Ways to consume the project. CLI / HuggingFace endpoint / hosted API are the next planned surfaces.

| Interface | Status | Description |
|---|---|---|
| Go binary (`bin/vad`) | DONE; NEEDS PACKAGE RELEASE | Self-contained ~59 MB binary (darwin/arm64) bundling every backend's weights and the ORT dylib via `go:embed`. Runs as a gRPC server (`vad.VoiceSegmentation`); selectable backend via `-backend` or `VADConfig` textproto. See [Quickstart](#quickstart). |
| Web server / standalone app (`cmd/basic-vad-web`) | DONE; NEEDS DEPLOYMENT| Three-process demo: vad gRPC + speax/audio gRPC + HTTP front. All VAD inference runs in the browser via onnxruntime-web; the demo server proxies `/fetch` for weights, `/upload` for arbitrary-container decoding, `/svg` for waveform rendering, and `/socket` for live mic → DetectStream bridging. See [Browser demo](#browser-demo-cmdbasic-vad-web). |
| CLI | SOON | One-shot `vad detect input.wav` style invocation (no server). |
| Client Libraries (Go, Python, C++, Java, etc.) | SOON | grpc (and http) client libraries for integration in other projects
| HuggingFace Inference Endpoint | SOON | Pushed images of each backend; pay-per-second hosted inference. |
| Public API | SOON | accretional-hosted endpoint |

## Quickstart

### Native (macOS arm64 / Linux x86_64 dev box)

```bash
bash setup.sh           # verifies deps; downloads ORT into third_party/ if missing
bash build-native.sh    # produces ./bin/vad (~59 MB, self-contained)

# Default backend (pyannote) on :50051:
./bin/vad

# Switch backend at startup:
./bin/vad -backend silero
./bin/vad -backend marblenet
./bin/vad -backend fsmn
./bin/vad -backend firered

# Or via a VADConfig textproto (preferred for non-default ports / overrides):
./bin/vad -config config/example.textproto
```

The ONNX Runtime dylib for the build target is embedded; no `ONNXRUNTIME_LIB` env var or `third_party/` needed at runtime. The on-disk fallback exists only for development.

### Container

```bash
docker build -t vad .                          # multi-stage build; ~70 MB final image
docker run -p 50051:50051 --rm vad             # default pyannote on :50051
docker run -p 50051:50051 --rm vad -backend silero
```

The image is `debian:bookworm-slim` + the single self-contained binary — no separate weights volume or ORT mount. Validate end-to-end with `bash test.sh` (runs the per-backend matrix in `tests/e2e/`).

### One-shot full test suite

```bash
bash test.sh   # go vet, all unit tests, in-process integration tests, the pkg-example pipeline,
               # and (if docker is reachable) docker build + per-backend container e2e + fetch tests
```

Layers that need Docker or ffmpeg skip gracefully if those tools aren't installed.

### Browser demo (`cmd/basic-vad-web`)

Multi-backend comparison demo where all VAD inference runs **in the browser** via `onnxruntime-web`. The server side is just a metadata + weights proxy:

```bash
./cmd/basic-vad-web/run.sh                    # defaults: vad :50051, audio :50052, demo :8080
./cmd/basic-vad-web/run.sh --backend silero   # change the server-side backend (only used for DetectStream)
./cmd/basic-vad-web/run.sh --audio-repo /path/to/speax/audio   # override the sibling audio checkout
```

Three processes: `bin/vad` (gRPC, weights + DetectStream), [`speax/audio`](https://github.com/accretional/speax/tree/main/audio) (gRPC `MediaConverter` for `/upload` decoding + `/svg` waveform rendering), and the demo HTTP server. The audio backend is optional — uploads and server-rendered waveforms degrade to in-browser equivalents when it's absent.

## Repo layout

- `pkg/vad/` — `Backend` interface + all 5 backend implementations.
- `weights/<backend>/` — bundled ONNX weights for each backend plus a `url.txt` sidecar (CDN-style URL the `Fetch` RPC returns when present, so clients can pull from GitHub raw instead of streaming via the server).
- `fbank/`, `melspec/` — log-Mel feature extractors (Kaldi and NeMo conventions).
- `internal/server/` — gRPC service (`Detect`, `DetectStream`, `Fetch`) implementation in Go
- `internal/embedded/` — `go:embed` indirection for the ORT dylib + per-backend weights.
- `cmd/vad/` — the gRPC server binary.
- `cmd/basic-vad-web/` — the browser-side multi-backend demo + HTTP front.
- `cmd/pkg-example/` — minimal CLI that drives the `pkg/vad` interface over `data/*.mp3` (encode → detect → segment → re-encode).
- `Dockerfile`, `build-native.sh`, `build.sh`, `setup.sh`, `test.sh`, `prep-embed.sh` — build / setup / test entry points (see header comments).
- `tests/{e2e,fetch,stream,basic-vad-web}/` — integration tests; `tests/e2e` and `tests/fetch` are Docker-driven (e2e includes the per-backend matrix).
- `docs/planning/original.md` — frozen snapshot of the original README development notes and plan.
- [`TODO.md`](TODO.md) — outstanding work (streaming RPC improvements, backend abstraction, CI).
