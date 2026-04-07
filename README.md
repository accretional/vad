# vad
Pyannote Segmentation 3.0 via ONNX and accretional/openvino-go as a remote grpc service (and weight server for transformers.js)

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

3. Define a vad.proto taking Audio inputs and Diarization outputs via a VoiceSegmentation grpc service using this and implement it using the package

4. Write an integration test against the grpc service in tests/e2e

5. Add a Fetch RPC to VoiceSegmentation that returns the model weights themselves, or optionally, a url configured as a CLI flag (if the URL is unconfigured, return the weights directly). Add a basic integration test validating the weights are returned properly in tests/fetch/

6. Add a cmd/basic-vad-web in which go/embed is used to serve index.html and accompanying css/js, and a dual grpc/http service (use essentially the same impl as https://github.com/accretional/katarche/tree/main/server) is used to implement VAD, for a basic web app that allows the user to select an audio file in their browser, send it to the VAD service, and visualize the segmented outputs (can be minimally processed, just the raw output in a table or something).

7. Use https://github.com/accretional/cdp-agent with https://github.com/accretional/chromerpc to test that server in tests/basic-vad-web by uploading an actual file. The agent should simply set up the automation by doing it manually at first, then using the structure of the web app/lower level chromerpc capabilities to programmatically upload the audio file without requiring vision in some automation like https://github.com/accretional/chromerpc/blob/main/automations/screenshot_site.textproto (but it can use a bash script and more than one automation textproto if necessary).

8. Implement a tests.sh that runs all the integration and unit tests. Make tests/basic-vad-web take screenshots 1s after the page loads and 1s after the final automation and have tests.sh save those to data/screenshots/YYY/MM/DD/basic-vad-web/start.png and data/screenshots/YYY/MM/DD/basic-vad-web/end.png

9. Implement a bench.sh that runs the CLI 1000 times sequentially on data/sorry-dave.mp3, and separately, one that starts the server once and invokes it in 1000 batches of sequentially on data/sorry-dave.mp3 with n=1,5,10,100 parallel requests from the client (the client should create a single reusable datastructure to make sure that this doesn't get bottlenecked by client io). Save the results to data/benchmarks/YYYY/MM/DD/cli.csv and data/benchmarks/YYYY/MM/DD/rpc.csv

10. Augment bench.sh with a 1000x fetch benchmark, save the results to data/benchmarks/YYYY/MM/DD/fetch.csv Also try to implement a container serving benchmark that deploys a container to google cloud run on instances with 2 vcpu and 4GB memory across 10 services (do this by uploading the container to artifact registry first, then doing all the deployments in parallel via gcloud, give each service the same prefix and a -1,5,10,100,1000 suffix and set their request concurrency to the same value), and sends each two subsequent vad requests once deployed, then records the deployment latency in data/benchmarks/YYYY/MM/DD/cloudrun-deploy-all.sh and data/benchmarks/YYYY/MM/DD/cloudrun-deploy-n.sh). Note that cloud run will have auto cold started these new services, so we cannot yet collect cold start benchmarks. Send each service 101 requests serially, through out the first request latency (it's not a true cold start but it may not be fully initialized either) and record the data in data/benchmarks/YYYY/MM/DD/serving-warm-all.csv and data/benchmarks/YYYY/MM/DD/serving-warm-n.csv. Now do the same but with the cleint sending 0.8*n concurrent requests (eg max 4 concurrent requests for the service with request concurrency of 5, max 8 for 10, max 80 for 100, max 800 for 1000) and save the data in data/benchmarks/YYYY/MM/DD/serving-throughput-n.csv. Then wait 15 minutes from the time the previous benchmarks ended. Finally, send each service one request and record the roundtrip latency for all in data/benchmarks/YYYY/MM/DD/serving-coldstart.csv

11. Copy cmd/basic-vad-web to a new dir cmd/transformers-vad-web and implement a ui that uses transformers.js to fetch the model on page load, load it via transformers.js into something that can run with webgpu, and process input through both the local and remote routes when provided. Add new tests that verify that the output is the same across both, and that use chromerpc/scraping to make sure that the user flow works in a real browser. Do something similar with screenshots and tests.sh, send them to data/screenshots/YYY/MM/DD/transformers-vad-web/start.png and data/screenshots/YYY/MM/DD/transformers-vad-web/end.png

12. Create a benchmark for coldstart/warm latency of webgpu serving of the model and save the data to data/benchmarks/YYYY/MM/DD/serving-webgpu-coldsdstart.csv and data/benchmarks/YYYY/MM/DD/serving-webgpu-warm.csv

13. Write a README.md about the project, go over .gitignore, go.mod, the Dockerfile, setup/test scripts, etc. to make sure everything is in a good shape too. Remove or address or log all TODOs and incomplete work, you can use NOTES.md to track info that doesn't belong in the user-facing README.md. Make sure in particular that the container-based build and serving are in good shape, as they are the primary ways we'll build and serve this project.
