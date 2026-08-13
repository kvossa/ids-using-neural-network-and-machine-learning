"""
Orquesta escenarios de laboratorio: captura por ventana, scripts de tráfico
y registro de ground truth (timestamps ISO 8601).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# Raíz del repositorio (…/ids)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.config_loader import load_lab_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestador de experimentos IDS")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiment" / "config" / "lab.json",
        help="Ruta a lab.json (o lab.yaml si tenéis PyYAML)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lista escenarios sin ejecutar captura ni scripts",
    )
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        print(
            f"No existe {cfg_path}. Copiad config/lab.example.json a config/lab.json",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = load_lab_config(cfg_path)
    cwd = resolve_path(ROOT, cfg.get("repo_root", "."))

    orch = cfg.get("orchestrator", {})
    results_dir = resolve_path(cwd, orch.get("results_dir", "experiment/results"))
    gt_path = resolve_path(cwd, orch.get("ground_truth_csv", "experiment/results/ground_truth.csv"))
    pcaps_dir = resolve_path(cwd, orch.get("pcaps_dir", "experiment/results/pcaps"))
    flows_csv = resolve_path(cwd, orch.get("flows_csv", "experiment/results/flows/session_flows.csv"))

    net = cfg.get("network", {})
    target = net.get("target_host", "127.0.0.1")
    benign_url = net.get("benign_url", f"http://{target}/")
    iface = net.get("capture_interface", "eth0")

    tcpdump_bin = cfg.get("tcpdump", {}).get("binary", "tcpdump")
    cfm = cfg.get("cicflowmeter", {})
    cfm_enabled = bool(cfm.get("enabled"))

    results_dir.mkdir(parents=True, exist_ok=True)
    pcaps_dir.mkdir(parents=True, exist_ok=True)
    flows_csv.parent.mkdir(parents=True, exist_ok=True)

    scenarios = orch.get("scenarios", [])
    if args.dry_run:
        for s in scenarios:
            print(f"[dry-run] {s.get('id')}: label={s.get('label')} script={s.get('script')}")
        return

    from experiment.capture.window_capture import capture_pcap, pcap_to_flows_dataframe

    fieldnames = ["scenario_id", "t_start", "t_end", "label", "notes", "pcap_path"]
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not gt_path.exists()
    gt_f = open(gt_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(gt_f, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    flow_frames = []
    errors: list[str] = []

    for sc in scenarios:
        sid = sc.get("id", "scenario")
        label = sc.get("label", "UNKNOWN")
        dur = float(sc.get("duration_sec", 30))
        script_rel = sc.get("script")
        do_capture = bool(sc.get("capture", True))
        notes = sc.get("notes", "")

        if not script_rel:
            errors.append(f"{sid}: sin script")
            continue

        script_path = resolve_path(cwd, script_rel)
        if not script_path.is_file():
            errors.append(f"{sid}: script no encontrado {script_path}")
            continue

        pcap_path = pcaps_dir / f"{sid}.pcap"
        env = os.environ.copy()
        env["TARGET_HOST"] = str(target)
        env["BENIGN_URL"] = str(benign_url)
        env["DURATION_SEC"] = str(int(dur))

        t_start = _utc_now()
        exc_capture: list[BaseException] = []
        exc_script: list[BaseException] = []

        def run_cap() -> None:
            try:
                if do_capture:
                    capture_pcap(iface, dur, pcap_path, tcpdump_bin=tcpdump_bin)
            except BaseException as e:
                exc_capture.append(e)

        def run_scr() -> None:
            try:
                subprocess.run(
                    ["bash", str(script_path)],
                    cwd=str(cwd),
                    env=env,
                    timeout=max(dur + 15.0, 60.0),
                    check=False,
                )
            except BaseException as e:
                exc_script.append(e)

        if do_capture:
            th_cap = threading.Thread(target=run_cap, daemon=True)
            th_scr = threading.Thread(target=run_scr, daemon=True)
            th_cap.start()
            th_scr.start()
            th_cap.join()
            th_scr.join()
        else:
            run_scr()

        t_end = _utc_now()

        if exc_capture:
            errors.append(f"{sid} capture: {exc_capture[0]}")
        if exc_script:
            errors.append(f"{sid} script: {exc_script[0]}")

        pcap_cell = str(pcap_path) if do_capture and pcap_path.exists() else ""
        writer.writerow(
            {
                "scenario_id": sid,
                "t_start": t_start,
                "t_end": t_end,
                "label": label,
                "notes": notes,
                "pcap_path": pcap_cell,
            }
        )
        gt_f.flush()

        if cfm_enabled and do_capture and pcap_path.exists() and pcap_path.stat().st_size > 0:
            try:
                df = pcap_to_flows_dataframe(
                    java_bin=cfm.get("java_bin", "java"),
                    jar_path=cfm.get("jar_path", ""),
                    pcap_path=pcap_path,
                    scenario_id=sid,
                )
                if df is not None and len(df) > 0:
                    flow_frames.append(df)
            except Exception as e:
                errors.append(f"{sid} CICFlowMeter: {e}")

    gt_f.close()

    if cfm_enabled and flow_frames:
        import pandas as pd

        merged = pd.concat(flow_frames, ignore_index=True)
        ts_col = cfg.get("inference", {}).get("timestamp_column", "Timestamp")
        if ts_col in merged.columns:
            merged["_sort_ts"] = pd.to_datetime(merged[ts_col], errors="coerce")
            merged = merged.sort_values("_sort_ts").drop(columns=["_sort_ts"])
        merged.to_csv(flows_csv, index=False)
        print(f"Flujos combinados: {flows_csv} ({len(merged)} filas)")
    elif not cfm_enabled:
        print("CICFlowMeter deshabilitado: generad el CSV de flujos manualmente o habilitad cicflowmeter.enabled.")

    if errors:
        print("Advertencias / errores:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Ground truth: {gt_path}")


if __name__ == "__main__":
    main()
