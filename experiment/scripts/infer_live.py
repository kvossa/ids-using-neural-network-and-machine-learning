"""
Live streaming inference: capture from interface → extract flows → model → predictions.csv

Usage:
  python experiment/scripts/infer_live.py --mode cic --interface eth0
  python experiment/scripts/infer_live.py --mode unsw --interface eth0 --output results/predictions.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Re-exec with user's Python when sudo loses venv
try:
    import numpy as np
except ImportError:
    sudo_user = os.environ.get("SUDO_USER", "freuer")
    user_home = f"/home/{sudo_user}"
    user_python = f"{user_home}/.pyenv/versions/3.9.25/bin/python"
    if not os.path.isfile(user_python):
        user_python = f"{user_home}/.pyenv/shims/python"
    if os.path.isfile(user_python):
        os.execv(user_python, [user_python] + sys.argv)
    sys.exit(f"Run with: sudo {user_python} " + " ".join(sys.argv))

from experiment.config_loader import load_lab_config


def _detect_interface() -> str:
    import subprocess
    try:
        result = subprocess.run(
            ["ip", "-br", "link"],
            capture_output=True, text=True, check=True,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "UP" and parts[0] != "lo":
                return parts[0]
    except Exception:
        pass
    return "eth0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live streaming inference")
    parser.add_argument(
        "--mode", choices=["cic", "unsw"], required=True,
        help="Dataset model to use for inference",
    )
    parser.add_argument(
        "--interface", default=None,
        help="Network interface (auto-detects if not specified)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Predictions CSV path (overrides config)",
    )
    parser.add_argument(
        "--log-features", default=None,
        help="Path to save preprocessed feature vectors for gap analysis",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Path to fine-tuned model directory (models/classification/fine_tuned/{mode})",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiment" / "config" / "lab.json",
        help="Lab config file",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    if cfg_path.is_file():
        cfg = load_lab_config(cfg_path)
        net = cfg.get("network", {})
        streaming = cfg.get("streaming", {})
        interface = args.interface or net.get("capture_interface") or _detect_interface()
        output = args.output or streaming.get("predictions_csv", f"results/predictions_{args.mode}.csv")
    else:
        interface = args.interface or _detect_interface()
        output = args.output or f"results/predictions_{args.mode}.csv"

    out_path = ROOT / output
    log_path = str(ROOT / args.log_features) if args.log_features else None

    if args.mode == "cic":
        from src.inference import StreamingCICInference
        model_dir = args.model_dir
        if model_dir is not None:
            stage1 = str(ROOT / model_dir / "stage1.keras")
            stage2 = str(ROOT / model_dir / "stage2.keras")
        else:
            stage1 = stage2 = None
        streamer = StreamingCICInference(
            interface=interface,
            predictions_csv=str(out_path),
            stage1_path=stage1,
            stage2_path=stage2,
        )
    else:
        from src.inference import StreamingUNSWInference
        model_path = str(ROOT / args.model_dir / "single_stage.keras") if args.model_dir else None
        streamer = StreamingUNSWInference(
            interface=interface,
            predictions_csv=str(out_path),
            model_path=model_path,
        )

    print(f"[infer_live] Mode={args.mode} Interface={interface} Output={out_path}")
    if log_path:
        print(f"[infer_live] Logging features to {log_path}")
    if args.model_dir:
        print(f"[infer_live] Using fine-tuned model: {args.model_dir}")
    streamer.run(log_features_path=log_path)


if __name__ == "__main__":
    main()
