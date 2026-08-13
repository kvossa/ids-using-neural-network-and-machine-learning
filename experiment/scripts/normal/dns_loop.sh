#!/usr/bin/env bash
# Consultas DNS repetidas (no requiere servidor DNS propio en TARGET_HOST).
set -euo pipefail
: "${TARGET_HOST:?Set TARGET_HOST (usado solo en el mensaje; resolución vía sistema)}"
DUR="${DURATION_SEC:-30}"
end=$((SECONDS + DUR))
echo "[dns_loop] resolviendo nombres durante ${DUR}s (TARGET_HOST=$TARGET_HOST)"
while (( SECONDS < end )); do
  dig +time=2 +tries=1 +short A example.com @1.1.1.1 >/dev/null 2>&1 || true
  dig +time=2 +tries=1 +short A cloudflare.com @8.8.8.8 >/dev/null 2>&1 || true
  sleep 0.5
done
