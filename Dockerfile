# Builds two binaries from one image:
#
#   1. "fat"  — embeds the ORT dylib AND every backend's weights. Suitable as
#               a standalone Linux executable: `docker buildx build --target
#               export --output type=local,dest=./out .` writes it to the host
#               as ./out/vad.
#
#   2. "slim" — embeds only the default backend (pyannote) plus the ORT dylib.
#               Other 4 backends load from /onnx/weights/<backend>/ at runtime,
#               which the runtime stage pre-populates by COPYing the full
#               weights tree. The runtime image stays small and lets you swap
#               weights via a bind mount without rebuilding the binary.
#
# Build targets:
#   docker buildx build --target runtime --load -t vad .         # slim container
#   docker buildx build --target export --output type=local,dest=./out .
#                                                                # fat host binary
#   docker buildx build --target runtime --platform linux/amd64,linux/arm64 \
#                       --tag $REGISTRY:tag --push .             # multi-arch manifest

FROM golang:1.26.1-bookworm AS builder

ARG MAIN_PKG=./cmd/vad
ARG ORT_VERSION=1.22.0

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download the ORT distribution into third_party/ where prep-embed.sh expects
# it. Auto-detect arch (linux-x64 / linux-aarch64).
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

COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true
COPY . .

# Materialise embed inputs (ORT .so + weights tree) into
# internal/embedded/{ort,weights}/ once; both build variants consume them.
ENV CGO_ENABLED=1
RUN ORT_PLATFORM_DIR=$(ls -d third_party/onnxruntime-linux-* | head -1) && \
    echo "export CGO_CFLAGS=\"-I/app/${ORT_PLATFORM_DIR}/include\"" > /etc/profile.d/cgo.sh && \
    echo "export CGO_LDFLAGS=\"-L/app/${ORT_PLATFORM_DIR}/lib\""    >> /etc/profile.d/cgo.sh && \
    chmod +x /etc/profile.d/cgo.sh && \
    . /etc/profile.d/cgo.sh && \
    bash prep-embed.sh && \
    go build -ldflags="-s -w"            -o /app/vad-fat  ${MAIN_PKG} && \
    go build -tags slim -ldflags="-s -w" -o /app/vad-slim ${MAIN_PKG} && \
    ls -lh /app/vad-fat /app/vad-slim

# ---------------------------------------------------------------------------
# Export stage: minimal scratch image whose sole purpose is to be the target of
# `docker buildx build --target export --output type=local,dest=./out`. The
# resulting host file is ./out/vad — the fat, self-contained Linux binary.
# ---------------------------------------------------------------------------
FROM scratch AS export
COPY --from=builder /app/vad-fat /vad

# ---------------------------------------------------------------------------
# Runtime stage: the actual container we ship. Slim binary + /onnx tree
# (weights for all 5 backends + the ORT dylib).
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# /onnx is the contract directory the slim binary expects:
#   /onnx/weights/<backend>/   on-disk weights for any backend not embedded
#                              (the default — pyannote — is also here so a
#                              bind-mount of /onnx can override even the
#                              embedded one).
#   /onnx/ort/                 spare copy of the ORT dylib; the slim binary
#                              materialises its embedded ORT into /onnx/ at
#                              startup, but ops folks doing image inspection
#                              may want to see the actual dylib path here too.
RUN mkdir -p /onnx/weights /onnx/ort
COPY --from=builder /app/vad-slim /server
COPY --from=builder /app/weights/ /onnx/weights/
COPY --from=builder /app/internal/embedded/ort/ /onnx/ort/

EXPOSE 50051
ENTRYPOINT ["/server"]
