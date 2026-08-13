#!/usr/bin/env bash
# Tráfico HTTP benigno repetible hacia el servidor de laboratorio.
# Variables de entorno (las define el orquestador):
#   TARGET_HOST, BENIGN_URL, DURATION_SEC
set -euo pipefail
: "${TARGET_HOST:?Set TARGET_HOST (e.g. 192.168.1.10)}"
URL="${BENIGN_URL:-http://${TARGET_HOST}/}"
DUR="${DURATION_SEC:-30}"
end=$((SECONDS + DUR))
echo "[http_benign] GET $URL durante ${DUR}s"
while (( SECONDS < end )); do
  curl -sS -m 5 -o /dev/null "$URL" || true
  sleep 1
done
