# vad
Pyannote Segmentation 3.0 via ONNX and accretional/openvino-go as a remote grpc service (and weight server for transformers.js)

## Development Rules

**IMPORTANT: No one-off commands.** All tests, builds, and setup steps MUST go through their respective scripts:
- **Testing**: Add to `test.sh`, then run `./test.sh`. Never run `go test` directly as a final validation.
- **Building**: Add to `build.sh` or the `Dockerfile`. Never run `go build` or `docker build` ad-hoc as a final step.
- **Setup**: Add to `setup.sh`. Never install dependencies or download artifacts outside of it.
- **Running demos**: Use the `run.sh` scripts in each `cmd/` directory.

If something needs to be tested or built, it belongs in a script. If it's not in a script, it doesn't count as tested.

## Plan

1. Check out the transformers.js usage of onnx-community/pyannote-segmentation-3.0 in https://huggingface.co/onnx-community/pyannote-segmentation-3.0#transformersjs-v3-usage. Note the original model is in https://huggingface.co/pyannote/segmentation-3.0 and we may need to end up implementing pipelining (though the onnx example docs suggest >10s in their output so it may include that?). Create a setup.sh that will be used to host all setup scripts for those who clone this repo and try to build and run what we are about to build. Start with something that checks if docker is installed. Then create a Dockerfile that takes golang-1.26.1-alpine3.23, adds onnx weights from weights/ to the filesystem under /weights, builds a golang binary against a parameterized main.go, then strips eveverything from the built binary into a container  from SCRATCH. Make sure that all works and that any necessary setup outside of the dockerfile is in the setup script

<details><summary>Implementation Notes</summary>

- `setup.sh`: checks Docker installation + daemon, downloads `model.onnx` (fp32, 5.99MB) from HuggingFace to `weights/`
- `Dockerfile`: multi-stage build from `golang:1.26.1-alpine3.23` → `scratch`. Uses `ARG MAIN_PKG=./cmd/vad` for parameterization. Builds with `CGO_ENABLED=0` and strips debug info with `-ldflags="-s -w"`. Final image contains only the binary, CA certs, and weights.
- `cmd/vad/main.go`: minimal placeholder, verified working via `docker build -t vad . && docker run --rm vad`
- Model info: input `[batch, 1, num_samples]` float32 (16kHz audio), output `[batch, num_frames, 7]` float32 logits
- Added `weights/` and built binary to `.gitignore`, created `.dockerignore`

</details>

2. Get golang onnx inference working using the weights in https://huggingface.co/onnx-community/pyannote-segmentation-3.0/tree/main/onnx (use similar approach to https:// pkg/embedder/cpu/onnx_embedder.go in github.com/accretional/rpcembed/ and ask Rithesh why this isn't using openvino-go) via a Go pkg, write unit tests and a cmd/pkg-example/main.go to demo it on data/sorry-dave.mp3 Use go/embed to bundle the weights into the go binary.

<details><summary>Implementation Notes</summary>

- `pkg/vad/`: pure inference package using `yalue/onnxruntime_go` v1.22.0 with ORT 1.22.0. Takes `[]float32` samples (16kHz mono), returns `[]Segment{Start, End, SpeakerID, Confidence}`. No filesystem access — caller provides audio data.
- Model outputs log-softmax probabilities across 7 classes: class 0 = no speech, classes 1-6 = speaker IDs. Post-processing applies `exp()` to get probabilities and thresholds at 0.5.
- `encode-to-16k.sh`: converts `data/*.mp3` to 16kHz mono float32 PCM (`.f32`) via ffmpeg CLI. TODO: replace with `ffmpeg-proto` audio_decode RPC.
- `internal/audio/load.go`: helper to load `.f32` files, used by tests and CLI.
- `cmd/pkg-example/main.go`: demo CLI, runs on `data/sorry-dave-16k.f32`, detects 16 segments across 2 speakers.
- All 3 unit tests pass (silence detection, sorry-dave inference, f32 loading).
- Dockerfile updated: alpine final stage with ONNX Runtime `.so`, CGO_ENABLED=1. See `RITHESH.md` for openvino-go discussion.
- `setup.sh` downloads both model weights and ONNX Runtime shared library for local dev.

</details>

3. Define a vad.proto taking Audio inputs and Diarization outputs via a VoiceSegmentation grpc service using this and implement it using the package

<details><summary>Implementation Notes</summary>

- `proto/vad.proto`: defines `VoiceSegmentation` service with `Detect(Audio) returns (Diarization)` RPC. Audio carries raw float32 PCM bytes + sample rate; Diarization returns repeated Segment (start, end, speaker_id, confidence) + duration.
- `proto/vadpb/`: generated Go code via `protoc --go_out` and `--go-grpc_out`.
- `internal/server/server.go`: implements the gRPC service, validates input (sample rate, non-empty, 4-byte aligned), converts bytes→float32, calls `model.ProcessAudio()`, maps results to proto Segments.
- `cmd/vad/main.go`: gRPC server binary with `-port`, `-model`, `-lib` flags. Graceful shutdown on SIGINT/SIGTERM.
- `setup.sh` updated: checks for protoc, protoc-gen-go, protoc-gen-go-grpc, and ffmpeg with install instructions.

</details>

4. Write an integration test against the grpc service in tests/e2e

<details><summary>Implementation Notes</summary>

- `tests/e2e/e2e_test.go`: builds Docker image, runs container with port mapping, connects gRPC client, runs 6 tests (silence, sorry-dave with multi-speaker validation, wake-me-up music rejection, invalid sample rate, empty audio, misaligned bytes).
- Dockerfile fixed: switched from alpine to `debian:bookworm-slim` (ONNX Runtime linux builds require glibc). Builder also uses `golang:1.26.1-bookworm`. Architecture auto-detected via `uname -m` for arm64/x64 ORT download.
- Container starts, serves gRPC on port 50051, and all tests pass end-to-end.

</details>

5. Add a Fetch RPC to VoiceSegmentation that returns the model weights themselves, or optionally, a url configured as a CLI flag (if the URL is unconfigured, return the weights directly). Add a basic integration test validating the weights are returned properly in tests/fetch/

<details><summary>Implementation Notes</summary>

- `proto/vad.proto`: added `Fetch(FetchRequest) returns (FetchResponse)` RPC with `oneof result { bytes weights; string url; }`.
- `internal/server/server.go`: implements Fetch — returns URL if `-weights-url` is configured, otherwise reads and returns raw model bytes from disk.
- `cmd/vad/main.go`: added `-weights-url` flag and bumped gRPC max message size to 32MB (model is ~6MB, exceeds default 4MB limit).
- `tests/fetch/fetch_test.go`: two tests — `TestFetchWeightsDirect` (no URL, expects ~6MB raw bytes) and `TestFetchWeightsURL` (with `-weights-url https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx`, expects URL string). Uses separate containers on ports 50052/50053.
- `test.sh`: orchestrates full test suite — go vet, unit tests, Docker build (`--no-cache`), e2e tests, fetch tests. Auto-detects ORT library path.
- `.dockerignore`: added `third_party/`, `debug/`, `*.f32` to reduce context size (142MB → 33MB).
- Dockerfile: switched to `golang:1.26.1-bookworm` builder + `debian:bookworm-slim` runtime (ONNX Runtime requires glibc). Architecture-aware ORT download via `uname -m`.

</details>

6. Add a cmd/basic-vad-web in which go/embed is used to serve index.html and accompanying css/js, and a dual grpc/http service (use essentially the same impl as https://github.com/accretional/katarche/tree/main/server) is used to implement VAD, for a basic web app that allows the user to select an audio file in their browser, send it to the VAD service, and visualize the segmented outputs (can be minimally processed, just the raw output in a table or something).

<details><summary>Implementation Notes</summary>

- `cmd/basic-vad-web/main.go`: dual gRPC/HTTP server on a single port using `h2c` (HTTP/2 cleartext). Routes `application/grpc` requests to gRPC server, everything else to HTTP mux. Serves embedded static files and `/api/detect` endpoint.
- `cmd/basic-vad-web/static/`: embedded via `go:embed` — `index.html`, `style.css`, `app.js`.
- `/api/detect`: accepts raw float32 PCM bytes via POST, returns JSON with segments array and duration. All audio decoding (MP3/WAV/OGG → 16kHz mono float32) is done client-side via Web Audio API.
- Client-side audio processing: `AudioContext.decodeAudioData()` handles any browser-supported format, `OfflineAudioContext` resamples to 16kHz mono. Segmented chunks are played back via in-browser WAV synthesis from float32 slices.
- UI features: file upload with drag-and-drop styling, results table with speaker color-coding, per-segment Play buttons, 25-second auto-close countdown timer (sticky header bar with red Pause/Resume button).
- `cmd/basic-vad-web/main_test.go`: 4 unit tests validating embedded static files (index.html content, CSS rules, JS endpoints, file count).
- `tests/basic-vad-web/web_test.go`: 6 e2e tests — builds binary, starts local server, validates HTTP responses for HTML/CSS/JS serving, 404 handling, and API inference on silence.
- `cmd/basic-vad-web/run.sh`: builds binary, starts server, validates all endpoints via curl, opens browser (skips in CI mode), waits for Ctrl+C.
- `test.sh` updated to include basic-vad-web unit and integration tests, plus run.sh scripts in CI mode.

</details>

6.5-BIG. **Anyserver: Generic gRPC+HTTP server framework with auto-doc, auto-linking, and service composition.**

The goal is to build a reusable, modular server framework in https://github.com/accretional/anyserver that any Go gRPC project (including this one) can use to automatically get: a dual gRPC/HTTP server, auto-generated godocs embedded and served over HTTP, grpc-gateway REST proxying with Swagger UI, source code browsing, and a polished index page. The framework should also support composing multiple gRPC services (from this repo and external ones like https://github.com/accretional/ffmpeg-proto and https://github.com/accretional/audio-visualizer) into a single server during build.

**User's vision:**
- Use https://github.com/accretional/godoc-gen (may require changes to that repo too) to automatically generate docs for the package and embed them in the binary, served over HTTP.
- Use https://github.com/accretional/audio-visualizer and/or https://github.com/accretional/ffmpeg-proto for better server-side audio processing and handling of more file formats.
- Better docs and more client-side functionality for managing output audio.
- This should be done in a modular, reusable way that can be used the same way in other repos: a way to partially or fully automate the integration of audio-visualizer and ffmpeg-proto gRPC services into the same gRPC server during build, and to generate the godocs of each, and their HTTP serving logic shim, and embed them all mostly automatically.
- https://github.com/accretional/katarche does some of this (gen_go.sh auto-generates main.go from `*_grpc.pb.go`, `server.Run()` callback pattern, HTTP reflection UI, proto import tooling via go_pull.sh).
- https://github.com/accretional/gluon handles Go→gRPC codegen (interface→proto, FullBootstrap, AST toolkit, proto compiler) but doesn't yet auto-integrate external gRPC services — it's close though.
- Katarche's patterns can be borrowed, and gluon can be extended, to achieve this.
- The vad repo stays the root for work until this project is done. Anyserver becomes the general server/doc/linker library.
- This repo's Dockerfile should eventually clone anyserver and inject this repo's gRPC service/package into its build, validating the new pattern works.
- Anyserver's default gRPC service should be called `Docs`, with RPCs: `GetSource(Path)` and `ListSource(Path)` (serving the repo's own `go:embed`-ed source code minus build artifacts but including .git), and `HTML(Path)` (serving auto-generated godoc HTML).
- Use https://github.com/grpc-ecosystem/grpc-gateway and protoc-gen-openapiv2 for REST proxying and OpenAPI spec generation.
- The index.html should serve Swagger UI, display the project README.md above it, use the repo name as the title/header, include basic metadata and links to code/docs, and use style.css for polish.

**Execution plan:**

Phase 1: Bootstrap anyserver (the generic framework)
- 6.5a. Set up anyserver repo with Go module, proto tooling, and basic structure: `server/` (dual gRPC/HTTP logic, borrowing from katarche's `server.Run()` callback pattern), `cmd/anyserver/main.go` (stubbed), `services.go` (top-level service registry that main.go uses to determine which gRPC services to start).
- 6.5b. Define `docs.proto` with service `Docs` containing `GetSource(SourceRequest) returns (SourceResponse)`, `ListSource(SourceRequest) returns (SourceListResponse)`, and `HTML(DocRequest) returns (DocResponse)`. Generate Go code.
- 6.5c. Implement `Docs` service: `go:embed` the repo's own source (minus build artifacts, including .git), serve via GetSource/ListSource. HTML serves auto-generated godoc output (initially placeholder, later from godoc-gen).
- 6.5d. Integrate grpc-gateway: add `google/api/annotations.proto` HTTP bindings to `docs.proto`, generate grpc-gateway reverse proxy and OpenAPI spec via protoc-gen-openapiv2. Wire the gateway mux into the HTTP server alongside static file serving.
- 6.5e. Build the index.html: render README.md content (embedded), Swagger UI pointed at the generated OpenAPI spec, repo name as title, metadata links to GitHub repo and godocs. Add style.css.
- 6.5f. Add a `tools/gen.sh` (inspired by katarche's `gen_go.sh`) that: runs protoc with go/grpc/gateway/openapi plugins, and optionally auto-generates the service registration in main.go by scanning `*_grpc.pb.go` files.

Phase 2: Service composition / linking pattern
- 6.5g. Design the "service injection" pattern: anyserver should accept external Go modules (like `github.com/accretional/vad`) that export a `Register(grpc.Server)` function or similar. The build process (via gen.sh or a go generate directive) clones/imports the external module, discovers its `*_grpc.pb.go` files, and auto-generates the registration calls in main.go.
- 6.5h. Make vad export a clean registration entry point: a `Register(s *grpc.Server, opts ...Option)` in eg `pkg/server/register.go` that wires up VoiceSegmentation with its model and config.
- 6.5i. Update vad's Dockerfile to clone anyserver, inject vad's service, and build — validating the composition pattern works end-to-end.

Phase 3: Docs and enhanced tooling
- 6.5j. Build or extend godoc-gen to produce static HTML from Go packages. Integrate its output into the Docs service's `HTML()` RPC. The build step runs godoc-gen on all packages and embeds the output.
- 6.5k. Define a pattern for integrating ffmpeg-proto and audio-visualizer as additional composable services — same Register() pattern, their protos get gateway bindings, their godocs get generated and embedded.
- 6.5l. Enhanced client-side audio management in basic-vad-web: download segmented audio, waveform visualization (via audio-visualizer if available, or a lightweight client-side fallback), better playback controls.

Phase 4: Validation
- 6.5m. Full test.sh validation: anyserver builds standalone, vad builds with anyserver composition, all existing vad tests still pass, Swagger UI and docs are accessible.

<details><summary>Implementation Notes</summary>

(To be filled in as work progresses)

</details>

7. Use https://github.com/accretional/cdp-agent with https://github.com/accretional/chromerpc to test that server in tests/basic-vad-web by uploading an actual file. The agent should simply set up the automation by doing it manually at first, then using the structure of the web app/lower level chromerpc capabilities to programmatically upload the audio file without requiring vision in some automation like https://github.com/accretional/chromerpc/blob/main/automations/screenshot_site.textproto (but it can use a bash script and more than one automation textproto if necessary).

8. Add screenshot automation to test.sh. Make tests/basic-vad-web take screenshots 1s after the page loads and 1s after the final automation and have test.sh save those to data/screenshots/YYY/MM/DD/basic-vad-web/start.png and data/screenshots/YYY/MM/DD/basic-vad-web/end.png

9. Implement a bench.sh that runs the CLI 1000 times sequentially on data/sorry-dave.mp3, and separately, one that starts the server once and invokes it in 1000 batches of sequentially on data/sorry-dave.mp3 with n=1,5,10,100 parallel requests from the client (the client should create a single reusable datastructure to make sure that this doesn't get bottlenecked by client io). Save the results to data/benchmarks/YYYY/MM/DD/cli.csv and data/benchmarks/YYYY/MM/DD/rpc.csv

10. Augment bench.sh with a 1000x fetch benchmark, save the results to data/benchmarks/YYYY/MM/DD/fetch.csv Also try to implement a container serving benchmark that deploys a container to google cloud run on instances with 2 vcpu and 4GB memory across 10 services (do this by uploading the container to artifact registry first, then doing all the deployments in parallel via gcloud, give each service the same prefix and a -1,5,10,100,1000 suffix and set their request concurrency to the same value), and sends each two subsequent vad requests once deployed, then records the deployment latency in data/benchmarks/YYYY/MM/DD/cloudrun-deploy-all.sh and data/benchmarks/YYYY/MM/DD/cloudrun-deploy-n.sh). Note that cloud run will have auto cold started these new services, so we cannot yet collect cold start benchmarks. Send each service 101 requests serially, through out the first request latency (it's not a true cold start but it may not be fully initialized either) and record the data in data/benchmarks/YYYY/MM/DD/serving-warm-all.csv and data/benchmarks/YYYY/MM/DD/serving-warm-n.csv. Now do the same but with the cleint sending 0.8*n concurrent requests (eg max 4 concurrent requests for the service with request concurrency of 5, max 8 for 10, max 80 for 100, max 800 for 1000) and save the data in data/benchmarks/YYYY/MM/DD/serving-throughput-n.csv. Then wait 15 minutes from the time the previous benchmarks ended. Finally, send each service one request and record the roundtrip latency for all in data/benchmarks/YYYY/MM/DD/serving-coldstart.csv

11. Copy cmd/basic-vad-web to a new dir cmd/transformers-vad-web and implement a ui that uses transformers.js to fetch the model on page load, load it via transformers.js into something that can run with webgpu, and process input through both the local and remote routes when provided. Add new tests that verify that the output is the same across both, and that use chromerpc/scraping to make sure that the user flow works in a real browser. Do something similar with screenshots and tests.sh, send them to data/screenshots/YYY/MM/DD/transformers-vad-web/start.png and data/screenshots/YYY/MM/DD/transformers-vad-web/end.png

12. Create a benchmark for coldstart/warm latency of webgpu serving of the model and save the data to data/benchmarks/YYYY/MM/DD/serving-webgpu-coldsdstart.csv and data/benchmarks/YYYY/MM/DD/serving-webgpu-warm.csv

13. Write a README.md about the project, go over .gitignore, go.mod, the Dockerfile, setup/test scripts, etc. to make sure everything is in a good shape too. Remove or address or log all TODOs and incomplete work, you can use NOTES.md to track info that doesn't belong in the user-facing README.md. Make sure in particular that the container-based build and serving are in good shape, as they are the primary ways we'll build and serve this project.
