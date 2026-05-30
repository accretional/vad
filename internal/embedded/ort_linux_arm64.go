//go:build linux && arm64

package embedded

import _ "embed"

//go:embed ort/libonnxruntime.so
var OrtLib []byte

const ortPlatformLabel = "linux/arm64"

func ortLibExt() string { return ".so" }
