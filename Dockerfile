FROM golang:1.26.1-alpine3.23 AS builder

ARG MAIN_PKG=./cmd/vad

RUN apk add --no-cache ca-certificates

WORKDIR /app

COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /app/server ${MAIN_PKG}

FROM scratch

COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /app/server /server
COPY weights/ /weights/

ENTRYPOINT ["/server"]
