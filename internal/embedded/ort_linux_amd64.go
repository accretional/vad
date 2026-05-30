//go:build linux && amd64

package embedded

import _ "embed"

//go:embed ort/libonnxruntime.so
var OrtLib []byte

const ortPlatformLabel = "linux/amd64"

func ortLibExt() string { return ".so" }
