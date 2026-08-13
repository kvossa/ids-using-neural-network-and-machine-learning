"""
Une ground_truth.csv y predictions.csv por solapamiento temporal.
Soporta formato binario (pred_attack) y formato grupo (prediction + group).
Para formato grupo, requiere --evaluate-groups y el dataset correspondiente.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_ts(s: str) -> Optional[datetime]:
    if not isinstance(s, str) or not s.strip():
        return None
    t = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t.to_pydatetime()


def overlaps(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    return a0 <= b1 and b0 <= a1


def is_benign_label(label: str, benign_set: set[str]) -> bool:
    return str(label).strip() in benign_set


def load_group_map(dataset: str) -> dict:
    if dataset == "CIC":
        from src.grouping.definitions import CIC_BRUTERARE_MAP
        return CIC_BRUTERARE_MAP
    elif dataset == "UNSW":
        from src.grouping.definitions import UNSW_CONFUSION_GROUP_MAP
        return dict(UNSW_CONFUSION_GROUP_MAP)
    return {}


def original_label_to_group(label: str, group_map: dict, normal_labels: set) -> str:
    label = str(label).strip()
    if is_benign_label(label, normal_labels):
        return None
    return group_map.get(label)


@dataclass
class WindowRow:
    w_start: datetime
    w_end: datetime
    pred_attack: int = 0
    prob_attack: float = 0.0
    prediction: str = ""
    confidence: float = 0.0
    group: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación laboratorio GT vs predicciones")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="JSON con métricas")
    parser.add_argument(
        "--benign-labels",
        nargs="*",
        default=["BENIGN", "Normal", "benign"],
        help="Etiquetas de ground truth consideradas negativas",
    )
    parser.add_argument(
        "--evaluate-groups", action="store_true",
        help="Evaluar a nivel de grupo (no binario ataque/benigno)",
    )
    parser.add_argument(
        "--dataset", choices=["CIC", "UNSW"], default=None,
        help="Dataset para mapear GT a grupos (requerido con --evaluate-groups)",
    )
    args = parser.parse_args()

    if args.evaluate_groups and not args.dataset:
        print("--dataset es requerido con --evaluate-groups", file=sys.stderr)
        sys.exit(1)

    benign_set = {str(x) for x in args.benign_labels}
    group_map = load_group_map(args.dataset) if args.evaluate_groups else {}
    all_groups: set[str] = set()

    gt_path = args.ground_truth.resolve()
    pred_path = args.predictions.resolve()
    if not gt_path.is_file() or not pred_path.is_file():
        print("Faltan archivos GT o predicciones.", file=sys.stderr)
        sys.exit(1)

    gt = pd.read_csv(gt_path)
    pr = pd.read_csv(pred_path)

    gt_rows = []
    for _, r in gt.iterrows():
        t0 = parse_ts(str(r.get("t_start", "")))
        t1 = parse_ts(str(r.get("t_end", "")))
        if t0 is None or t1 is None:
            continue
        label = str(r.get("label", ""))
        gt_group = None
        if args.evaluate_groups:
            gt_group = original_label_to_group(label, group_map, benign_set)
            if gt_group is not None:
                all_groups.add(gt_group)
        gt_rows.append({
            "scenario_id": r.get("scenario_id", ""),
            "t_start": t0,
            "t_end": t1,
            "label": label,
            "is_attack": not is_benign_label(label, benign_set),
            "group": gt_group,
        })

    # Detect format: group-based (has 'prediction' column) or binary (has 'pred_attack')
    has_group_col = "prediction" in pr.columns
    has_instant_ts = "window_t_start" not in pr.columns and "_timestamp" in pr.columns
    windows: list[WindowRow] = []
    for _, r in pr.iterrows():
        if has_instant_ts:
            t0 = parse_ts(str(r.get("_timestamp", "")))
            t1 = t0
        else:
            t0 = parse_ts(str(r.get("window_t_start", "")))
            t1 = parse_ts(str(r.get("window_t_end", "")))
        if t0 is None or t1 is None:
            continue
        w = WindowRow(w_start=t0, w_end=t1)
        if has_group_col:
            w.prediction = str(r.get("prediction", ""))
            w.confidence = float(r.get("confidence", 0.0))
            w.group = str(r.get("group", ""))
            w.pred_attack = 0 if is_benign_label(w.prediction, benign_set) else 1
            w.prob_attack = w.confidence if w.pred_attack else 1.0 - w.confidence
        else:
            w.pred_attack = int(r.get("pred_attack", 0))
            w.prob_attack = float(r.get("prob_attack", 0.0))
        windows.append(w)

    if not windows:
        print("No hay ventanas válidas en predictions.", file=sys.stderr)
        sys.exit(1)

    y_true_bin = []
    y_pred_bin = []
    y_true_group = []
    y_pred_group = []

    for w in windows:
        attack_intervals = [gr for gr in gt_rows if gr["is_attack"] and overlaps(w.w_start, w.w_end, gr["t_start"], gr["t_end"])]
        benign_intervals = [gr for gr in gt_rows if not gr["is_attack"] and overlaps(w.w_start, w.w_end, gr["t_start"], gr["t_end"])]

        if attack_intervals:
            y_true_bin.append(1)
        elif benign_intervals:
            y_true_bin.append(0)
        else:
            y_true_bin.append(0)

        y_pred_bin.append(1 if w.pred_attack else 0)

        if args.evaluate_groups:
            if attack_intervals:
                true_group = attack_intervals[0]["group"]
                y_true_group.append(true_group if true_group else "BENIGN")
                all_groups.add(true_group) if true_group else None
            else:
                y_true_group.append("BENIGN")
            pred_group = w.group if w.group else "BENIGN"
            if is_benign_label(pred_group, benign_set):
                pred_group = "BENIGN"
            y_pred_group.append(pred_group)

    y_true_a = np.array(y_true_bin, dtype=int)
    y_pred_a = np.array(y_pred_bin, dtype=int)

    f1 = float(f1_score(y_true_a, y_pred_a, zero_division=0))
    prec = float(precision_score(y_true_a, y_pred_a, zero_division=0))
    rec = float(recall_score(y_true_a, y_pred_a, zero_division=0))

    # Falsos positivos en regiones solo benignas
    fp_windows = 0
    benign_only_windows = 0
    for w in windows:
        attack_ovl = any(gr["is_attack"] and overlaps(w.w_start, w.w_end, gr["t_start"], gr["t_end"]) for gr in gt_rows)
        benign_ovl = any((not gr["is_attack"]) and overlaps(w.w_start, w.w_end, gr["t_start"], gr["t_end"]) for gr in gt_rows)
        if benign_ovl and not attack_ovl:
            benign_only_windows += 1
            if w.pred_attack:
                fp_windows += 1

    fp_rate_benign = float(fp_windows / benign_only_windows) if benign_only_windows else 0.0

    # Latencia
    latencies_sec: list[float] = []
    for gr in gt_rows:
        if not gr["is_attack"]:
            continue
        t_start = gr["t_start"]
        first_alert: Optional[datetime] = None
        for w in windows:
            if not w.pred_attack:
                continue
            if overlaps(w.w_start, w.w_end, gr["t_start"], gr["t_end"]):
                first_alert = w.w_start
                break
        if first_alert is not None:
            latencies_sec.append(max(0.0, (first_alert - t_start).total_seconds()))

    # Mixtos
    mixed_windows = sum(
        1 for w in windows
        if any((not gr["is_attack"]) and overlaps(w.w_start, w.w_end, gr["t_start"], gr["t_end"]) for gr in gt_rows)
        and any(gr["is_attack"] and overlaps(w.w_start, w.w_end, gr["t_start"], gr["t_end"]) for gr in gt_rows)
    )

    report: dict = {
        "windows_evaluated": len(windows),
        "f1_window_level": f1,
        "precision_window_level": prec,
        "recall_window_level": rec,
        "false_positive_windows_in_benign_only_regions": fp_windows,
        "benign_only_windows": benign_only_windows,
        "false_positive_rate_benign_only_regions": fp_rate_benign,
        "attack_scenario_latency_seconds": {
            "count": len(latencies_sec),
            "mean": float(np.mean(latencies_sec)) if latencies_sec else None,
            "min": float(np.min(latencies_sec)) if latencies_sec else None,
            "max": float(np.max(latencies_sec)) if latencies_sec else None,
        },
        "mixed_overlap_windows": mixed_windows,
    }

    # Group-level metrics
    if args.evaluate_groups:
        group_metrics = {}
        if y_true_group:
            all_groups = sorted(set(y_true_group + y_pred_group))
            for g in all_groups:
                y_t = np.array([1 if x == g else 0 for x in y_true_group])
                y_p = np.array([1 if x == g else 0 for x in y_pred_group])
                g_f1 = float(f1_score(y_t, y_p, zero_division=0))
                g_prec = float(precision_score(y_t, y_p, zero_division=0))
                g_rec = float(recall_score(y_t, y_p, zero_division=0))
                if g not in ("BENIGN",):
                    group_metrics[g] = {
                        "f1": g_f1,
                        "precision": g_prec,
                        "recall": g_rec,
                        "count": int(y_t.sum()),
                    }
            group_labels = list(group_metrics.keys())
            if group_labels:
                macro_f1 = float(np.mean([group_metrics[g]["f1"] for g in group_labels]))
                report["group_level"] = {
                    "macro_f1": macro_f1,
                    "per_group": group_metrics,
                }
        report["confusion_groups_note"] = (
            "True labels mapped to groups using the dataset's group map. "
            "BENIGN/Normal predictions excluded from per-group metrics."
        )

    out = args.output
    if out:
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Métricas: {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
