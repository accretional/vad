//go:build slim

package main

// onDiskWeightsRoot in slim builds points at the canonical location where
// the container's Dockerfile COPYs the full weights tree. Only the default
// backend (pyannote) is embedded; the other 4 backends are loaded from here
// on demand.
const onDiskWeightsRoot = "/onnx/weights"

// embedMaterializeRoot is where the embedded ORT dylib and the default
// model's extracted weights land at startup. /onnx is the stable, mountable
// location the container reserves for runtime files — keeping the
// materialized dylib alongside the on-disk weights tree means everything
// the server needs to read at runtime lives under one directory.
const embedMaterializeRoot = "/onnx"
