#!/usr/bin/env bash
# Escaneo de puertos (SYN) reproducible. Solo en redes de laboratorio autorizadas.
set -euo pipefail
: "${TARGET_HOST:?Set TARGET_HOST}"
# Perfil conservador: top puertos, sin versión agresiva
exec nmap -sS -T4 --top-ports 100 --open -Pn "$TARGET_HOST"
