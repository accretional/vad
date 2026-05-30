package embedded

// PlatformLabel is a human-readable string identifying which platform's ORT
// dylib (if any) is embedded in this binary. Used in startup logging /
// diagnostics; never used for code-path selection.
func PlatformLabel() string { return ortPlatformLabel }

// OrtBytes returns the embedded ORT dylib size in bytes. Useful for
// startup-time diagnostics (e.g. log if it diverges from a discovered local
// install).
func OrtBytes() int { return len(OrtLib) }
