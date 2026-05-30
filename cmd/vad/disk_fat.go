//go:build !slim

package main

// onDiskWeightsRoot is the directory the server checks first for per-backend
// weights. Disk-first means dev edits to weights/<backend>/ take effect
// without rebuilding; the embed only kicks in for backends not on disk.
const onDiskWeightsRoot = "weights"
