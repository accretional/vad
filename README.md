# vad

gRPC voice-activity-detection service with five pluggable backends. All backends speak the same `vad.VoiceSegmentation` proto (`Detect` unary + `DetectStream` bidi streaming + `Fetch` for weights/URL handoff), so clients are backend-agnostic.

A single self-contained binary (`bin/vad`, ~59 MB on darwin/arm64) bundles every backend's ONNX weights and the matching ONNX Runtime dylib via `go:embed`. No separate setup beyond `bash build-native.sh` once you have Go and ffmpeg.

## Backends

| `-backend` | Model | Source | Size | Speaker IDs | Notes |
|---|---|---|---|---|---|
| `pyannote` (default) | Pyannote Segmentation 3.0 | [`onnx-community/pyannote-segmentation-3.0`](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) | ~6 MB | yes (up to 3) | Full diarization. |
| `fsmn` | FunASR FSMN-VAD | [`funasr/fsmn-vad`](https://huggingface.co/funasr/fsmn-vad) (PyTorch → ONNX; export script at [accretional/speax/benchmarks/vad](https://github.com/accretional/speax/tree/main/benchmarks/vad)) | 1.6 MB | no | Smallest. Chinese-trained but works on English; verify accuracy on your audio. |
| `firered` | FireRedTeam DFSMN-VAD | [`FireRedTeam/FireRedVAD`](https://huggingface.co/FireRedTeam/FireRedVAD) (PyTorch → ONNX) | 2.3 MB | no | Closest in segment quality to pyannote on the bundled sample clips. |
| `silero` | Silero VAD | [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad) | 1.3 MB | no | Chunked-state streaming model. |
| `marblenet` | NVIDIA MarbleNet (Frame VAD Multilingual v2.0) | [`nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) | 1 MB | no | Fastest in our benches. Multilingual. NeMo log-mel features (`melspec/`). |

Pure-Go log-Mel features live in two parity-tested packages:
- `fbank/` — Kaldi-style (Povey/Hamming window, HTK mel, snip_edges); used by FSMN + FireRed. Parity-tested against `kaldi_native_fbank`. See [`fbank/README.md`](fbank/README.md).
- `melspec/` — NeMo-style (Hann window, Slaney mel, center reflection padding, `log(mel + 2⁻²⁴)`); used by MarbleNet. Parity-tested against NeMo's `AudioToMelSpectrogramPreprocessor`.

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
