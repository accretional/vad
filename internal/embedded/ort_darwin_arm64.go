//go:build darwin && arm64

package embedded

import _ "embed"

//go:embed ort/libonnxruntime.dylib
var OrtLib []byte

const ortPlatformLabel = "darwin/arm64"

func ortLibExt() string { return ".dylib" }
