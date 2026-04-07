FROM golang:1.26.1-alpine3.23 AS builder

ARG MAIN_PKG=./cmd/vad
ARG ORT_VERSION=1.22.0

RUN apk add --no-cache ca-certificates gcc musl-dev curl

# Download ONNX Runtime for Linux x64
RUN curl -sL -o /tmp/ort.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" && \
    tar xzf /tmp/ort.tgz -C /usr/local && \
    rm /tmp/ort.tgz && \
    ln -s /usr/local/onnxruntime-linux-x64-${ORT_VERSION} /usr/local/onnxruntime

ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/onnxruntime/include"
ENV CGO_LDFLAGS="-L/usr/local/onnxruntime/lib"
ENV LD_LIBRARY_PATH=/usr/local/onnxruntime/lib

WORKDIR /app

COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true

COPY . .

RUN go build -ldflags="-s -w" -o /app/server ${MAIN_PKG}

FROM alpine:3.23

RUN apk add --no-cache ca-certificates

# Copy ONNX Runtime shared library
COPY --from=builder /usr/local/onnxruntime/lib/ /tmp/ort-lib/
RUN cp /tmp/ort-lib/libonnxruntime.so* /usr/local/lib/ 2>/dev/null; \
    cp /tmp/ort-lib/libonnxruntime.1*.so /usr/local/lib/ 2>/dev/null; \
    rm -rf /tmp/ort-lib && \
    ldconfig /usr/local/lib 2>/dev/null || true

COPY --from=builder /app/server /server
COPY weights/ /weights/

ENV ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so

ENTRYPOINT ["/server"]
