//go:build slim

package embedded

import "embed"

// Weights in slim builds embeds only the default backend (pyannote). The
// remaining backends MUST be present on disk under the configured weights
// root (typically /onnx/weights/ inside the container; see
// cmd/vad/disk_slim.go for the constant). ResolveWeights / WeightsBytes /
// WeightsURL fall back to disk when the embed lookup misses, so callers
// don't need to branch on BuildMode.
//
//go:embed all:weights/pyannote
var Weights embed.FS

// BuildMode reports which compile-time variant produced this binary.
// "slim" reduces binary size by ~10 MB at the cost of needing the
// non-default model weights present on disk at runtime.
const BuildMode = "slim"
