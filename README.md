# vad

gRPC voice-activity-detection service with five pluggable backends. All backends speak the same `vad.VoiceSegmentation` proto (`Detect` unary + `DetectStream` bidi streaming + `Fetch` for weights/URL handoff), so clients are backend-agnostic.

A single self-contained binary (`bin/vad`, ~59 MB on darwin/arm64) bundles every backend's ONNX weights and the matching ONNX Runtime dylib via `go:embed`. No separate setup beyond `bash build-native.sh` once you have Go and ffmpeg.

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
| CPU | ✅ shipping | Default ORT CPU EP. All 5 backends run end-to-end. |
| GPU (Metal) | 🚧 planned | ORT 1.22+ ships CoreML EP for darwin/arm64; not wired into `NewDynamicAdvancedSession` calls yet (see `pkg/vad/*.go`). |
| NPU (Apple Neural Engine) | 🚧 planned | Same CoreML EP — opt-in flag would route eligible ops to the ANE. |
| Browser (transformers.js / onnxruntime-web) | ✅ shipping | Demo runs all 5 models WASM-side via Web Workers; `Fetch` RPC hands out CDN URLs from per-backend `url.txt` sidecars. |
| Go package | ✅ shipping | `import "github.com/accretional/vad/pkg/vad"` — see `pkg/vad/`. |
| gRPC service | ✅ shipping | `./bin/vad` exposes `vad.VoiceSegmentation` (`Detect`, `DetectStream`, `Fetch`). |
| Standalone app (demo) | ✅ shipping | `./cmd/basic-vad-web/run.sh` brings up vad + audio + demo HTTP. |

### Linux (x86_64)

| Target | Status | Notes |
|---|---|---|
| CPU | 🟡 coming soon | ORT linux-x64 dylib is wired into `internal/embedded/` build path; needs CI green-light. |
| GPU (CUDA / TensorRT) | 🟡 coming soon | — |
| Browser | 🟡 coming soon | Same WASM bundle; needs Linux-host validation. |
| Go package | 🟡 coming soon | — |
| gRPC service | 🟡 coming soon | — |
| Standalone app (demo) | 🟡 coming soon | — |

### [REDACTED] — *Beyond the Edge, Ambient Inference*

| Target | Status | Notes |
|---|---|---|
| ▣▣▣ Core | ⟁ classified ⟁ | Operates in regimes where conventional EPs do not apply. |
| ▣▣▣ Coprocessor | ⟁ classified ⟁ | Power envelope: *negligible*. Latency floor: *sub-perceptual*. |
| Browser | ⟂ N/A ⟂ | The browser is downstream. |
| Go package | ⟁ classified ⟁ | Surface area: undisclosed. |
| gRPC service | ⟂ N/A ⟂ | Out-of-band transport. |
| Ambient runtime | ⟁ classified ⟁ | Already on. |

## Interfaces

Ways to consume the project. CLI / HuggingFace endpoint / hosted API are the next planned surfaces.

| Interface | Status | Description |
|---|---|---|
| Go binary (`bin/vad`) | ✅ shipping | Self-contained ~59 MB binary (darwin/arm64) bundling every backend's weights and the ORT dylib via `go:embed`. Runs as a gRPC server (`vad.VoiceSegmentation`); selectable backend via `-backend` or `VADConfig` textproto. See [Quickstart](#quickstart). |
| Web server / standalone app (`cmd/basic-vad-web`) | ✅ shipping | Three-process demo: vad gRPC + speax/audio gRPC + HTTP front. All VAD inference runs in the browser via onnxruntime-web; the demo server proxies `/fetch` for weights, `/upload` for arbitrary-container decoding, `/svg` for waveform rendering, and `/socket` for live mic → DetectStream bridging. See [Browser demo](#browser-demo-cmdbasic-vad-web). |
| CLI | 🟡 coming soon | One-shot `vad detect input.wav` style invocation (no server). |
| HuggingFace Inference Endpoint | 🟡 coming soon | Pushed images of each backend; pay-per-second hosted inference. |
| Public API | 🟡 coming soon | accretional-hosted endpoint with auth + per-model routing. |

## Quickstart

```bash
bash setup.sh           # verifies deps; the bundled binary already has weights + ORT embedded
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
- `internal/server/` — gRPC service (`Detect`, `DetectStream`, `Fetch`).
- `internal/embedded/` — `go:embed` indirection for the ORT dylib + per-backend weights.
- `fbank/`, `melspec/` — log-Mel feature extractors (Kaldi and NeMo conventions).
- `cmd/vad/` — the gRPC server binary.
- `cmd/basic-vad-web/` — the browser-side multi-backend demo + HTTP front.
- `weights/<backend>/` — bundled ONNX weights for each backend plus a `url.txt` sidecar (CDN-style URL the `Fetch` RPC returns when present, so clients can pull from GitHub raw instead of streaming via the server).
- `tests/{e2e,fetch,stream,basic-vad-web}/` — integration tests; `tests/e2e` and `tests/fetch` are Docker-driven.
- `docs/planning/original.md` — frozen snapshot of the original README development notes and plan.
- [`TODO.md`](TODO.md) — outstanding work (streaming RPC improvements, backend abstraction, CI).
