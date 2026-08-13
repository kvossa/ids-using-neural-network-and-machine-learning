"""
Inferencia batch: CSV de flujos → preprocessing → ventanas → modelo → predictions.csv

Soporta dataset CIC (two-stage) y UNSW (single-stage multiclass).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiment.config_loader import load_lab_config
from src.inference.cic import CICInference
from src.inference.unsw import UNSWInference


def resolve_path(cwd: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (cwd / q).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia batch sobre CSV de flujos")
    parser.add_argument("--config", type=Path, default=ROOT / "experiment" / "config" / "lab.json")
    parser.add_argument("--flows-csv", type=Path, required=True, help="CSV de flujos")
    parser.add_argument("--output", type=Path, default=None, help="Salida predictions.csv")
    parser.add_argument(
        "--dataset", choices=["CIC", "UNSW"], default=None,
        help="Forzar dataset (por defecto usa el config)",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        print(f"Falta {cfg_path}. Copiad lab.example.json a lab.json", file=sys.stderr)
        sys.exit(1)

    cfg = load_lab_config(cfg_path)
    cwd = resolve_path(ROOT, cfg.get("repo_root", "."))

    art = cfg.get("artifacts", {})
    dataset = (args.dataset or art.get("dataset", "CIC")).upper()
    orch = cfg.get("orchestrator", {})
    out_pred = args.output or resolve_path(cwd, orch.get("predictions_csv", "results/predictions.csv"))
    out_pred = out_pred.resolve()
    out_pred.parent.mkdir(parents=True, exist_ok=True)

    flows_path = args.flows_csv.resolve()
    if not flows_path.is_file():
        print(f"No existe flows CSV: {flows_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(flows_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    print(f"Dataset: {dataset}, Flows: {len(df)}, Output: {out_pred}")

    ts_col = None
    df_ts = None
    inf = cfg.get("inference", {})

    if dataset == "CIC":
        # CIC pipeline: two-stage
        drop_cols = list(inf.get("drop_columns_after_load", [])) + ["Label"]
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
        ts_col = inf.get("timestamp_column", "Timestamp")
        detector = CICInference()
    else:
        # UNSW pipeline: single-stage multiclass
        drop_cols = list(inf.get("drop_columns_after_load", []))
        for c in drop_cols + ["id", "attack_cat", "label"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        ts_col = inf.get("timestamp_column", "timestamp")
        detector = UNSWInference()

    if ts_col and ts_col in df.columns:
        df_ts = pd.to_datetime(df[ts_col], errors="coerce")
        order = df_ts.argsort()
        df = df.iloc[order].reset_index(drop=True)
        df_ts = df_ts.iloc[order].reset_index(drop=True)

    results = detector.predict_raw(df)

    ws = detector.WINDOW_SIZE
    rows_out = []
    for i, r in enumerate(results):
        t_start = t_end = None
        if df_ts is not None and i + ws - 1 < len(df_ts):
            t_start = df_ts.iloc[i]
            t_end = df_ts.iloc[i + ws - 1]
        rows_out.append({
            "window_index": i,
            "window_t_start": t_start.isoformat() if t_start is not None and not pd.isna(t_start) else "",
            "window_t_end": t_end.isoformat() if t_end is not None and not pd.isna(t_end) else "",
            "prediction": r.get("prediction", ""),
            "confidence": r.get("confidence", 0.0),
            "group": r.get("group", ""),
            "stage1_result": r.get("stage1_result", ""),
            "stage1_confidence": r.get("stage1_confidence", 0.0),
            "source_flows_csv": str(flows_path),
        })

    pd.DataFrame(rows_out).to_csv(out_pred, index=False)
    print(f"Predicciones: {out_pred} ({len(rows_out)} ventanas)")


if __name__ == "__main__":
    main()
