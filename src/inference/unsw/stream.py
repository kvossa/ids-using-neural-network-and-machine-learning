import os
import shutil
import subprocess
import threading
import time
from contextlib import redirect_stdout
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from ..base import load_preprocessor
from .batch import UNSWInference
from ..streaming import FlowBuffer, PredictionCSVWriter, FeatureLogger
from .features import (
    UNSWFlow,
    ConnectionStateTable,
    _parse_tshark_line,
    TSHARK_FIELDS,
)


class StreamingUNSWInference:
    DROP_COLUMNS = ["attack_cat", "label", "id"]

    def __init__(
        self,
        interface: str,
        predictions_csv: str = "predictions.csv",
        model_path: Optional[str] = None,
    ):
        self.interface = interface
        if model_path is not None:
            self.unsw = UNSWInference(model_path=model_path)
        else:
            self.unsw = UNSWInference()
        self.preprocessor = load_preprocessor(UNSWInference.PREPROCESSOR_PATH)
        self.writer = PredictionCSVWriter(predictions_csv)
        self.ct_table = ConnectionStateTable()

        self.buffer = FlowBuffer(
            window_size=UNSWInference.WINDOW_SIZE,
            on_prediction=self._predict_window,
            n_features=193,
        )
        self._flows: Dict[tuple, UNSWFlow] = {}
        self._lock = threading.Lock()
        self._tshark_proc: Optional[subprocess.Popen] = None
        self._completed_flows = 0
        self._flow_count_for_reset = 0
        self._last_progress = 0
        self.feature_logger = None

    def _predict_window(self, X_ae: np.ndarray, X_seq: np.ndarray) -> Dict:
        result = self.unsw._predict_model(X_ae, X_seq)[0]
        return result

    def _process_packet(self, pkt_dict: Dict):
        if pkt_dict["proto"] not in (6, 17):
            return
        src, dst, sport, dport, proto = (
            pkt_dict["src_ip"], pkt_dict["dst_ip"],
            pkt_dict["src_port"], pkt_dict["dst_port"],
            pkt_dict["proto"],
        )
        fwd_key = (src, dst, sport, dport, proto)
        rev_key = (dst, src, dport, sport, proto)

        with self._lock:
            flow = self._flows.get(fwd_key) or self._flows.get(rev_key)
            if flow is None:
                flow = UNSWFlow(pkt_dict)
                self._flows[fwd_key] = flow
            else:
                flow.add_packet(pkt_dict)

            completed = []
            now = pkt_dict["time"]
            for key, f in list(self._flows.items()):
                if f.is_completed(now):
                    completed.append((key, f))
                    del self._flows[key]

        for key, f in completed:
            features = f.to_dict()
            self.ct_table.update_features(f, features)
            self.ct_table.add_flow(f, features)
            self._feed_to_buffer(features)

        if completed:
            self._completed_flows += len(completed)
            self._flow_count_for_reset += len(completed)
            if self._flow_count_for_reset >= 500:
                self.ct_table.reset()
                self._flow_count_for_reset = 0
            if self._completed_flows - self._last_progress >= 100:
                print(f"[UNSW] Completed flows: {self._completed_flows} | Buffer: {len(self.buffer.buffer)}/{UNSWInference.WINDOW_SIZE} | CT resets: {self._completed_flows // 500}", flush=True)
                self._last_progress = self._completed_flows

    def _feed_to_buffer(self, features: Dict):
        df = pd.DataFrame([features])
        try:
            with redirect_stdout(open(os.devnull, "w")):
                X_proc = self.preprocessor.transform(df)
        except Exception as e:
            print(f"[UNSW] Preprocessor error: {e}", flush=True)
            return
        feat = np.asarray(X_proc, dtype=np.float32).ravel()
        if self.feature_logger is not None:
            self.feature_logger.log(feat)
        result = self.buffer.add(feat)
        if result is not None:
            print(f"[UNSW] Prediction: {result.get('prediction', '?')} (conf={result.get('confidence', 0):.3f})", flush=True)
            self.writer.write(result)

    def _tshark_reader(self):
        field_str = " ".join(f"-e {f}" for f in TSHARK_FIELDS)
        cmd = (
            f"tshark -i {self.interface} -l -T fields {field_str} "
            f"-E occurrence=f "
            f"ip"
        )
        self._tshark_proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=None,
            bufsize=1, text=True,
        )
        for line in self._tshark_proc.stdout:
            pkt = _parse_tshark_line(line)
            if pkt is not None:
                self._process_packet(pkt)

    def run(self, log_features_path: Optional[str] = None):
        if not shutil.which("tshark"):
            print("[UNSW] ERROR: tshark not found. Install with: sudo pacman -S tshark", flush=True)
            return
        if log_features_path is not None:
            self.feature_logger = FeatureLogger(log_features_path, 193)
        print(f"[UNSW] Starting live capture on {self.interface}...", flush=True)
        try:
            self._tshark_reader()
        except KeyboardInterrupt:
            print("\n[UNSW] Stopping...")
        finally:
            if self._tshark_proc:
                self._tshark_proc.terminate()
                self._tshark_proc.wait(timeout=5)
            now = time.time()
            with self._lock:
                remaining = list(self._flows.values())
                self._flows.clear()
            for f in remaining:
                features = f.to_dict()
                self.ct_table.update_features(f, features)
                self._feed_to_buffer(features)
            self.writer.close()
            if self.feature_logger is not None:
                self.feature_logger.close()
