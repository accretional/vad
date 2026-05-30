// release-review serves an HTML review page for a release in progress and
// blocks on the operator's decision (ACCEPT or CANCEL). Designed to be the
// last gate before release.sh creates the zip and pushes — the operator
// sees provenance, artifact list, and per-artifact validation results,
// then clicks one button.
//
// Reads release-logs/<sha>/metadata.json for the build context, serves
// review.html (embedded), and exits 0 on accept / 1 on cancel.
//
// Usage:
//
//	release-review -logs release-logs/abc1234
//
// The page also serves the contents of the logs dir under /logs/ so the
// operator can click into each individual log file.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	_ "embed"
)

//go:embed review.html
var reviewTpl string

// Metadata is the JSON shape release.sh writes to release-logs/<sha>/metadata.json.
type Metadata struct {
	Tag          string       `json:"tag"`
	GitSHA       string       `json:"git_sha"`
	GitBranch    string       `json:"git_branch"`
	Date         string       `json:"date"`
	BuildHost    string       `json:"build_host"`
	BuildOSArch  string       `json:"build_os_arch"`
	BuildUser    string       `json:"build_user"`
	BuildStarted string       `json:"build_started"`
	Artifacts    []Artifact   `json:"artifacts"`
	Validations  []Validation `json:"validations"`
	Logs         []LogFile    `json:"logs"`
}

type Artifact struct {
	Name   string `json:"name"`
	Path   string `json:"path"`
	Size   string `json:"size"`
	SHA256 string `json:"sha256"`
}

type Validation struct {
	Name    string `json:"name"`
	Port    int    `json:"port"`
	OK      bool   `json:"ok"`
	LogPath string `json:"log_path"`
}

type LogFile struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

func (m Metadata) HasFailures() bool {
	for _, v := range m.Validations {
		if !v.OK {
			return true
		}
	}
	return false
}

func main() {
	logsDir := flag.String("logs", "", "release-logs/<sha>/ directory (required)")
	addr := flag.String("addr", "127.0.0.1:0", "review server listen address; :0 picks a free port")
	openBrowser := flag.Bool("open", true, "open the review page in the default browser")
	flag.Parse()

	if *logsDir == "" {
		log.Fatal("-logs is required")
	}
	metaPath := filepath.Join(*logsDir, "metadata.json")
	data, err := os.ReadFile(metaPath)
	if err != nil {
		log.Fatalf("read %s: %v", metaPath, err)
	}
	var meta Metadata
	if err := json.Unmarshal(data, &meta); err != nil {
		log.Fatalf("parse %s: %v", metaPath, err)
	}

	tpl, err := template.New("review").Parse(reviewTpl)
	if err != nil {
		log.Fatalf("template: %v", err)
	}

	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("listen %s: %v", *addr, err)
	}
	defer lis.Close()
	url := "http://" + lis.Addr().String()

	decision := make(chan string, 1)

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := tpl.Execute(w, meta); err != nil {
			log.Printf("template execute: %v", err)
		}
	})
	mux.Handle("/logs/", http.StripPrefix("/logs/", http.FileServer(http.Dir(*logsDir))))
	mux.HandleFunc("/decision", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Verdict string `json:"verdict"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		switch body.Verdict {
		case "accept", "reject":
			fmt.Fprintln(w, "OK")
			select {
			case decision <- body.Verdict:
			default:
			}
		default:
			http.Error(w, "unknown verdict", http.StatusBadRequest)
		}
	})

	srv := &http.Server{Handler: mux}
	go func() {
		if err := srv.Serve(lis); err != nil && err != http.ErrServerClosed {
			log.Printf("serve: %v", err)
		}
	}()

	fmt.Fprintf(os.Stderr, "Review server: %s\n", url)
	if *openBrowser {
		var opener string
		switch runtime.GOOS {
		case "darwin":
			opener = "open"
		case "linux":
			opener = "xdg-open"
		}
		if opener != "" {
			_ = exec.Command(opener, url).Run()
		}
	}
	fmt.Fprintln(os.Stderr, "Waiting for ACCEPT or CANCEL …")

	verdict := <-decision

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)

	if verdict == "accept" {
		fmt.Fprintln(os.Stderr, "Accepted.")
		os.Exit(0)
	}
	fmt.Fprintln(os.Stderr, "Cancelled.")
	os.Exit(1)
}
