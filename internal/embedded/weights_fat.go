//go:build !slim

package embedded

import "embed"

// Weights is the embedded weights tree. Files mirror the on-disk weights/
// layout under the repo root: weights/<backend>/model.onnx, etc. In the
// default ("fat") build every backend's weights are baked in via go:embed.
//
// `all:` prefix opts in to embedding files whose names begin with `.` or `_`,
// which we don't really need but is harmless and future-proof.
//
//go:embed all:weights
var Weights embed.FS

// BuildMode reports which compile-time variant produced this binary —
// useful for startup logs / observability. "fat" embeds every backend;
// "slim" (build with `-tags slim`) embeds only the default.
const BuildMode = "fat"
