#!/usr/bin/env bash
# PLANTILLA — DoS puede afectar disponibilidad. Solo con autorización explícita.
set -euo pipefail
: "${TARGET_HOST:?Set TARGET_HOST}"
echo "[dos_hping] Plantilla: no envía tráfico. Editar para usar hping3 con límites acordados." >&2
exit 0
