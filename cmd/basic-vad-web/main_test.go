package main

import (
	"io/fs"
	"strings"
	"testing"
)

func TestStaticFSContainsIndex(t *testing.T) {
	data, err := staticFS.ReadFile("static/index.html")
	if err != nil {
		t.Fatalf("failed to read static/index.html: %v", err)
	}
	html := string(data)

	if !strings.Contains(html, "<title>") {
		t.Error("index.html missing <title> tag")
	}
	if !strings.Contains(html, "Voice Activity Detection") {
		t.Error("index.html missing expected title text")
	}
	if !strings.Contains(html, "style.css") {
		t.Error("index.html missing style.css reference")
	}
	if !strings.Contains(html, "app.js") {
		t.Error("index.html missing app.js reference")
	}
	t.Logf("index.html: %d bytes", len(data))
}

func TestStaticFSContainsCSS(t *testing.T) {
	data, err := staticFS.ReadFile("static/style.css")
	if err != nil {
		t.Fatalf("failed to read static/style.css: %v", err)
	}
	css := string(data)

	if !strings.Contains(css, ".container") {
		t.Error("style.css missing .container rule")
	}
	if !strings.Contains(css, "table") {
		t.Error("style.css missing table rule")
	}
	t.Logf("style.css: %d bytes", len(data))
}

func TestStaticFSContainsJS(t *testing.T) {
	data, err := staticFS.ReadFile("static/app.js")
	if err != nil {
		t.Fatalf("failed to read static/app.js: %v", err)
	}
	js := string(data)

	if !strings.Contains(js, "/api/detect") {
		t.Error("app.js missing /api/detect endpoint")
	}
	if !strings.Contains(js, "displayResults") {
		t.Error("app.js missing displayResults function")
	}
	t.Logf("app.js: %d bytes", len(data))
}

func TestStaticFSFileCount(t *testing.T) {
	var count int
	fs.WalkDir(staticFS, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			count++
			t.Logf("  %s", path)
		}
		return nil
	})
	if count != 3 {
		t.Errorf("expected 3 static files (index.html, style.css, app.js), got %d", count)
	}
}
