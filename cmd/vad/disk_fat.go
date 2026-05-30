//go:build !slim

package main

// onDiskWeightsRoot is the conventional directory the server checks for
// per-backend on-disk weights (used as a fallback when something isn't
// embedded; e.g. a model added after the binary was built). In the default
// "fat" build every backend is embedded, so this is mostly a dev convenience.
const onDiskWeightsRoot = "weights"

// embedMaterializeRoot is the parent dir for the embedded-ORT dylib and
// embedded-weights extraction at startup. "" means use os.CreateTemp's
// default (/tmp on Unix). Slim builds redirect to /onnx — see disk_slim.go.
const embedMaterializeRoot = ""
