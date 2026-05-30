//go:build !(darwin && arm64) && !(linux && amd64)

package embedded

// OrtLib is empty on build targets that don't have a bundled dylib. The
// server falls back to discoverOnnxRuntime / the ONNXRUNTIME_LIB env var /
// the -lib flag to find a system install. Add more //go:embed files in
// ort_<goos>_<goarch>.go to ship additional platforms.
var OrtLib []byte

const ortPlatformLabel = "unsupported"

func ortLibExt() string { return "" }
