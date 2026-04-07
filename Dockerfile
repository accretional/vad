FROM golang:1.26.1-bookworm AS builder

ARG MAIN_PKG=./cmd/vad
ARG ORT_VERSION=1.22.0
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime for the build architecture
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        ORT_PLATFORM="linux-aarch64"; \
    else \
        ORT_PLATFORM="linux-x64"; \
    fi && \
    curl -sL -o /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz" && \
    tar xzf /tmp/ort.tgz -C /usr/local && \
    rm /tmp/ort.tgz && \
    ln -s /usr/local/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION} /usr/local/onnxruntime

ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/onnxruntime/include"
ENV CGO_LDFLAGS="-L/usr/local/onnxruntime/lib"
ENV LD_LIBRARY_PATH=/usr/local/onnxruntime/lib

WORKDIR /app

COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true

COPY . .

RUN go build -ldflags="-s -w" -o /app/server ${MAIN_PKG}

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime shared library
COPY --from=builder /usr/local/onnxruntime/lib/ /tmp/ort-lib/
RUN cp /tmp/ort-lib/libonnxruntime.so* /usr/local/lib/ 2>/dev/null; \
    rm -rf /tmp/ort-lib && \
    ldconfig

COPY --from=builder /app/server /server
COPY weights/ /weights/

ENV ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so

ENTRYPOINT ["/server"]
