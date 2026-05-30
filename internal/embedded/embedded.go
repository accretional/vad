// Package embedded ships the ONNX Runtime shared library and the per-backend
// model weights bundled into the vad binary at compile time. Build steps in
// build.sh materialize third_party/ and weights/ into ort/ and weights/
// subdirs here right before `go build`, so the binary is self-contained.
//
// Two surfaces:
//   - OrtLib: bytes of libonnxruntime.{dylib,so} for the build target's
//     platform. Empty []byte if not embedded for this platform.
//   - Weights: an embed.FS rooted at weights/, mirroring the on-disk layout
//     (weights/<backend>/<file>). Helper functions resolve a path against
//     this FS or fall back to the on-disk weights/ directory if the file is
//     present there (so users can drop in updated weights without rebuilding).
package embedded

import (
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// Weights is the embedded weights tree. Files mirror the on-disk weights/
// layout under the repo root: weights/<backend>/model.onnx, etc.
//
// `all:` prefix opts in to embedding files whose names begin with `.` or `_`,
// which we don't really need but is harmless and future-proof.
//
//go:embed all:weights
var Weights embed.FS

// HasEmbeddedDylib reports whether OrtLib is populated for this build target.
// Defined in the per-platform files (ort_darwin_arm64.go etc.); the fallback
// in ort_unsupported.go returns false.
func HasEmbeddedDylib() bool { return len(OrtLib) > 0 }

// MaterializeDylib writes the embedded ORT shared library bytes to a unique
// temp file and returns the path, suitable for InitONNXRuntime. Caller is
// responsible for removing the file when done (typically at process exit).
//
// Returns an error if the build target has no embedded dylib.
func MaterializeDylib() (string, error) {
	if !HasEmbeddedDylib() {
		return "", fmt.Errorf("no embedded ORT dylib for this build target")
	}
	tmp, err := os.CreateTemp("", "libonnxruntime-*"+ortLibExt())
	if err != nil {
		return "", fmt.Errorf("create temp dylib: %w", err)
	}
	defer tmp.Close()
	if _, err := tmp.Write(OrtLib); err != nil {
		os.Remove(tmp.Name())
		return "", fmt.Errorf("write temp dylib: %w", err)
	}
	if err := os.Chmod(tmp.Name(), 0o755); err != nil {
		return "", fmt.Errorf("chmod temp dylib: %w", err)
	}
	return tmp.Name(), nil
}

// ResolveWeights returns a directory containing the requested backend's
// weights, ready to pass to a backend constructor.
//
// Resolution order (embedded-first):
//  1. If the embedded `weights/<backend>/model.onnx` exists, materialize the
//     embedded tree to a temp dir and return that. This is the canonical path
//     since the binary ships with weights at build time.
//  2. Otherwise, if `<onDiskRoot>/<backend>/model.onnx` exists on disk, return
//     that directory. This is the escape hatch for backends added after the
//     binary was built (drop new weights under weights/ to extend without
//     recompiling — useful when the VADModel enum has been extended but the
//     binary release lags).
//
// `onDiskRoot` is the conventional weights/ directory (e.g. "weights" when
// running from the repo root). Pass "" to skip the disk fallback entirely.
//
// `tempDir` is non-empty only when materialization happened; caller should
// `os.RemoveAll(tempDir)` at shutdown.
func ResolveWeights(onDiskRoot, backend string) (dir, tempDir string, err error) {
	if backend == "" {
		return "", "", errors.New("backend cannot be empty")
	}
	embedRoot := "weights/" + backend
	if _, statErr := fs.Stat(Weights, embedRoot+"/model.onnx"); statErr == nil {
		tmpDir, mkErr := os.MkdirTemp("", "vad-weights-"+backend+"-*")
		if mkErr != nil {
			return "", "", fmt.Errorf("mkdir temp: %w", mkErr)
		}
		if matErr := materializeDir(Weights, embedRoot, tmpDir); matErr != nil {
			os.RemoveAll(tmpDir)
			return "", "", matErr
		}
		return tmpDir, tmpDir, nil
	}
	if onDiskRoot != "" {
		candidate := filepath.Join(onDiskRoot, backend)
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			if _, err := os.Stat(filepath.Join(candidate, "model.onnx")); err == nil {
				return candidate, "", nil
			}
		}
	}
	return "", "", fmt.Errorf("no weights for backend %q (not embedded, not on disk at %q)", backend, onDiskRoot)
}

// WeightsBytes returns the raw ONNX bytes for the requested backend's primary
// model file. Embedded-first, with disk fallback at `<onDiskRoot>/<backend>/model.onnx`.
// Used by the Fetch RPC to serve weights to clients.
func WeightsBytes(onDiskRoot, backend string) ([]byte, error) {
	embedRoot := "weights/" + backend + "/model.onnx"
	if data, err := fs.ReadFile(Weights, embedRoot); err == nil {
		return data, nil
	}
	if onDiskRoot != "" {
		path := filepath.Join(onDiskRoot, backend, "model.onnx")
		if data, err := os.ReadFile(path); err == nil {
			return data, nil
		}
	}
	return nil, fmt.Errorf("no weights for backend %q", backend)
}

// AvailableBackends returns the subset of `candidates` for which weights are
// available (embedded preferred; disk fallback). Used at startup to decide
// which backends are loadable.
func AvailableBackends(onDiskRoot string, candidates []string) []string {
	out := make([]string, 0, len(candidates))
	for _, b := range candidates {
		if _, err := fs.Stat(Weights, "weights/"+b+"/model.onnx"); err == nil {
			out = append(out, b)
			continue
		}
		if onDiskRoot != "" {
			if _, err := os.Stat(filepath.Join(onDiskRoot, b, "model.onnx")); err == nil {
				out = append(out, b)
			}
		}
	}
	return out
}

func materializeDir(src embed.FS, root, dst string) error {
	return fs.WalkDir(src, root, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := fs.ReadFile(src, p)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
}
