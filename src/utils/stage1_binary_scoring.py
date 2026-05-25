"""Shared Stage-1 (binary) attack score transform: optional isotonic calibration on P(attack)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


CALIBRATOR_FILENAME = "calibrator_iso.pkl"


def load_threshold_config(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def apply_stage1_attack_score(
    raw_attack_prob: np.ndarray,
    stage1_dir: Path | str,
    threshold_data: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Map raw model P(attack) column to the score space used when the threshold was chosen.

    ``raw_attack_prob`` is typically ``predict(...)['classification'][:, 1]``.
    """
    stage1_dir = Path(stage1_dir)
    p = np.asarray(raw_attack_prob, dtype=np.float64).ravel()
    if threshold_data is None:
        with open(stage1_dir / "threshold.json") as f:
            threshold_data = json.load(f)
    cal = threshold_data.get("calibration", "none")
    if cal == "isotonic":
        iso = joblib.load(stage1_dir / CALIBRATOR_FILENAME)
        return np.asarray(iso.predict(p), dtype=np.float64)
    return p
