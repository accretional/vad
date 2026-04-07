# Notes for Rithesh

## Why onnxruntime_go instead of openvino-go?

This project uses [yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go) for ONNX inference rather than [accretional/openvino-go](https://github.com/accretional/openvino-go). Here's why:

### Both use CGO

openvino-go was presumably created to provide a pure-Go inference path, but it actually uses CGO too — it wraps a custom C++ library (`libopenvino_wrapper.so`) via `internal/cgo/`. So both libraries have the same CGO dependency story, and neither solves the `CGO_ENABLED=0` / scratch container problem on its own.

### Cross-platform support

openvino-go ships a prebuilt wrapper for **Linux amd64 only**. Other platforms require building the C++ wrapper from source (CMake + OpenVINO dev headers). onnxruntime_go supports Linux, macOS (including Apple Silicon via CoreML), and Windows out of the box.

### Hardware flexibility

- **onnxruntime_go**: CPU, CUDA (NVIDIA), TensorRT, CoreML (Apple), DirectML (AMD), and more
- **openvino-go**: Intel CPU, Intel GPU, Intel NPU only

For a project that may deploy to Cloud Run (likely AMD/Intel CPUs) but also needs to build and test on macOS, onnxruntime_go is the pragmatic choice.

### Maturity

onnxruntime_go has ~600 stars, 77 forks, and 3+ years of active use. openvino-go has 1 star and is ~2 months old. The API surface of openvino-go looks reasonable but hasn't seen community validation yet.

### rpcembed precedent

rpcembed already uses onnxruntime_go successfully in `pkg/embedder/cpu/onnx_embedder.go`:

```go
session, err := ort.NewAdvancedSession(modelPath, inputNames, outputNames, inputs, outputs, nil)
// ...
session.Run()
```

This pattern works and is proven in the accretional ecosystem. Interestingly, rpcembed has a `DownloadOpenVINOModel` helper but doesn't use openvino-go for actual inference.

### Question

Was openvino-go intended to eventually be pure Go (no CGO)? If so, that would be a compelling reason to revisit this choice once it matures. For now, since both need CGO anyway, onnxruntime_go is the better-supported option.

### Container implications

Since we need CGO, the Dockerfile can't use a pure `scratch` final image with a statically-linked binary. Options:
1. Use `alpine` as the final stage and install the ONNX Runtime shared library
2. Use a distroless base image with the shared library copied in
3. Copy just the required `.so` files into `scratch` (fragile but minimal)

We're going with option 1 for reliability.
