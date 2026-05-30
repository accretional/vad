//go:build slim

package main

// onDiskWeightsRoot in slim builds points at the canonical location where
// the container's Dockerfile COPYs the full weights tree. All 5 backends
// load from here. The slim binary still embeds the default backend
// (pyannote) as a fallback for when it's extracted from the container and
// run on a host without /onnx/weights/ — in the normal container path the
// embed is never read.
const onDiskWeightsRoot = "/onnx/weights"
