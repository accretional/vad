# Builds the `vad` gRPC server (or any -arg-overridden cmd/<pkg>) as a single
# self-contained binary. All backend weights and the libonnxruntime.so for the
# build target are bundled via go:embed (internal/embedded/{ort,weights}/);
# the runtime image carries no extra files beyond libc + ca-certs.
#
# Build:  docker build -t vad .
# Run:    docker run --rm -p 50051:50051 vad

FROM golang:1.26.1-bookworm AS builder

ARG MAIN_PKG=./cmd/vad
ARG ORT_VERSION=1.22.0

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 1: download the ORT distribution into third_party/ where prep-embed.sh
# expects it, then materialise the embed artifacts. The platform is auto-
# detected from the build host's arch (linux-x64 today; linux-aarch64
# tomorrow when we wire ort_linux_arm64.go).
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  ORT_PLATFORM=linux-x64 ;; \
        aarch64) ORT_PLATFORM=linux-aarch64 ;; \
        *) echo "unsupported arch: $ARCH" >&2; exit 1 ;; \
    esac && \
    mkdir -p third_party && \
    curl -sL -o /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz" && \
    tar xzf /tmp/ort.tgz -C third_party && \
    rm /tmp/ort.tgz

# Stage 2: source + module cache.
COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true
COPY . .

# Stage 3: materialise embed inputs into internal/embedded/{ort,weights}/
# and build. CGO_ENABLED=1 because the ORT binding is a CGO wrapper; we
# point CGO_CFLAGS/LDFLAGS at the third_party/ tree we just unpacked.
ENV CGO_ENABLED=1
RUN ORT_PLATFORM_DIR=$(ls -d third_party/onnxruntime-linux-* | head -1) && \
    export CGO_CFLAGS="-I/app/${ORT_PLATFORM_DIR}/include" && \
    export CGO_LDFLAGS="-L/app/${ORT_PLATFORM_DIR}/lib" && \
    bash prep-embed.sh && \
    go build -ldflags="-s -w" -o /app/server ${MAIN_PKG}

# Runtime: minimal Debian (libc only). The binary embeds everything it needs.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/server /server
EXPOSE 50051
ENTRYPOINT ["/server"]
