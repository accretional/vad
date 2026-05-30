// Package embedded ships the ONNX Runtime shared library and the per-backend
// model weights bundled into the vad binary at compile time. Build steps in
// prep-embed.sh materialize third_party/ and weights/ into ort/ and weights/
// subdirs here right before `go build`, so the binary is self-contained.
//
// Two surfaces:
//   - OrtLib: bytes of libonnxruntime.{dylib,so} for the build target's
//     platform. Empty []byte if not embedded for this platform.
//   - Weights: an embed.FS rooted at weights/, mirroring the on-disk layout
//     (weights/<backend>/<file>). What's actually embedded depends on the
//     build tag (see weights_fat.go vs weights_slim.go).
//
// Resolution is DISK-FIRST throughout. Reasoning: in the slim-container
// deployment, /onnx/weights/<backend>/ is the canonical location populated
// by the Dockerfile, and we don't want to materialize embedded copies
// alongside it. In the dev / standalone-binary case, the embed kicks in
// only when nothing is on disk — which is exactly when it's needed.
package embedded

import (
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Weights is declared in weights_fat.go (default: embeds everything) or
// weights_slim.go (build with `-tags slim`: only the default backend).
// See those files for what's actually embedded. The `embed` import is kept
// here so dependent code keeps compiling even when both variant files
// would somehow be excluded by tooling.
var _ = embed.FS{}

// HasEmbeddedDylib reports whether OrtLib is populated for this build target.
// Defined in the per-platform files (ort_darwin_arm64.go etc.); the fallback
// in ort_unsupported.go returns false.
func HasEmbeddedDylib() bool { return len(OrtLib) > 0 }

// MaterializeDylib writes the embedded ORT shared library bytes to a unique
// temp file and returns the path, suitable for InitONNXRuntime. Caller is
// responsible for removing the file when done (typically at process exit).
//
// Used only as a fallback when no ORT dylib is found on disk; the normal
// container path picks /onnx/ort/<libname> via discoverOnnxRuntime.
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
// Resolution order (DISK-FIRST):
//  1. If `<onDiskRoot>/<backend>/model.onnx` exists, return that directory
//     as-is. This is the normal path in the slim container — the Dockerfile
//     COPYs the full weights tree to /onnx/weights/.
//  2. Otherwise, if the embedded `weights/<backend>/model.onnx` exists,
//     materialize the embedded tree to a temp dir and return that. This is
//     the fallback for standalone binaries that have no weights tree on disk.
//
// `onDiskRoot` is the conventional weights/ directory (e.g. "weights" when
// running from the repo root, or "/onnx/weights" in slim container builds).
// Pass "" to skip the disk check entirely.
//
// `tempDir` is non-empty only when materialization happened; caller should
// `os.RemoveAll(tempDir)` at shutdown.
func ResolveWeights(onDiskRoot, backend string) (dir, tempDir string, err error) {
	if backend == "" {
		return "", "", errors.New("backend cannot be empty")
	}
	if onDiskRoot != "" {
		candidate := filepath.Join(onDiskRoot, backend)
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			if _, err := os.Stat(filepath.Join(candidate, "model.onnx")); err == nil {
				return candidate, "", nil
			}
		}
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
	return "", "", fmt.Errorf("no weights for backend %q (not on disk at %q, not embedded)", backend, onDiskRoot)
}

// WeightsURL returns the CDN URL stored in the backend's url.txt sidecar, if
// any. Resolution mirrors WeightsBytes (disk-first):
//
//  1. If `<onDiskRoot>/<backend>/url.txt` exists, return its trimmed contents.
//  2. Otherwise, if the embedded `weights/<backend>/url.txt` exists, return
//     its trimmed contents.
//  3. Otherwise return ("", false).
//
// Used by the Fetch RPC to redirect clients to a CDN download (much smaller
// payload through the gRPC pipe — clients fetch the .onnx directly from the
// returned HTTPS URL). The convention is one line per file: a single URL.
func WeightsURL(onDiskRoot, backend string) (string, bool) {
	if onDiskRoot != "" {
		path := filepath.Join(onDiskRoot, backend, "url.txt")
		if data, err := os.ReadFile(path); err == nil {
			if s := firstNonEmptyLine(string(data)); s != "" {
				return s, true
			}
		}
	}
	embedPath := "weights/" + backend + "/url.txt"
	if data, err := fs.ReadFile(Weights, embedPath); err == nil {
		if s := firstNonEmptyLine(string(data)); s != "" {
			return s, true
		}
	}
	return "", false
}

// firstNonEmptyLine returns the first non-empty whitespace-trimmed line of s,
// or "" if there is none. url.txt should contain exactly one URL; this
// tolerates trailing newlines / accidental whitespace.
func firstNonEmptyLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(line); t != "" {
			return t
		}
	}
	return ""
}

// WeightsBytes returns the raw ONNX bytes for the requested backend's primary
// model file. Disk-first, with embed fallback. Used by the Fetch RPC to
// serve weights to clients.
func WeightsBytes(onDiskRoot, backend string) ([]byte, error) {
	if onDiskRoot != "" {
		path := filepath.Join(onDiskRoot, backend, "model.onnx")
		if data, err := os.ReadFile(path); err == nil {
			return data, nil
		}
	}
	embedRoot := "weights/" + backend + "/model.onnx"
	if data, err := fs.ReadFile(Weights, embedRoot); err == nil {
		return data, nil
	}
	return nil, fmt.Errorf("no weights for backend %q", backend)
}

// AvailableBackends returns the subset of `candidates` for which weights are
// available (disk-first, embed fallback). Used at startup to decide which
// backends are loadable.
func AvailableBackends(onDiskRoot string, candidates []string) []string {
	out := make([]string, 0, len(candidates))
	for _, b := range candidates {
		if onDiskRoot != "" {
			if _, err := os.Stat(filepath.Join(onDiskRoot, b, "model.onnx")); err == nil {
				out = append(out, b)
				continue
			}
		}
		if _, err := fs.Stat(Weights, "weights/"+b+"/model.onnx"); err == nil {
			out = append(out, b)
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
