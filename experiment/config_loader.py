"""Carga de lab.json / lab.yaml sin depender de PyYAML para JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_lab_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    text = path.read_text(encoding="utf-8")
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Para archivos .yaml/.yml instalad PyYAML (pip install pyyaml) "
                "o usad lab.json (ver config/lab.example.json)."
            ) from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config inválida (se esperaba objeto): {path}")
    return data
