import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional
from pathlib import Path

from scapy.sendrecv import AsyncSniffer
from scapy.config import conf as scapy_conf

from src.cicflowmeter.flow_session import FlowSession
from src.cicflowmeter.writer import OutputWriter
from src.cicflowmeter.sniffer import _start_periodic_gc
from src.cicflowmeter.constants import EXPIRED_UPDATE, PACKETS_PER_GC

from src.inference.base import load_preprocessor
from src.inference.cic import CICInference
from src.inference.streaming import FlowBuffer, PredictionCSVWriter, FeatureLogger
from src.config import CIC_STAGE1, CIC_STAGE2

CIC_COLUMN_MAP = {
    "protocol": "Protocol",
    "flow_duration": "Flow Duration",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Fwd Packets Length Total",
    "totlen_bwd_pkts": "Bwd Packets Length Total",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "pkt_len_max": "Packet Length Max",
    "pkt_len_min": "Packet Length Min",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_seg_size_min": "Fwd Seg Size Min",
    "fwd_act_data_pkts": "Fwd Act Data Packets",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "flow_iat_std": "Flow IAT Std",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Avg Packet Size",
    "init_fwd_win_byts": "Init Fwd Win Bytes",
    "init_bwd_win_byts": "Init Bwd Win Bytes",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "cwr_flag_count": "CWE Flag Count",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
}


def remap_cic_columns(data: dict) -> dict:
    mapped = {}
    for py_key, cic_key in CIC_COLUMN_MAP.items():
        val = data.get(py_key)
        if val is not None:
            mapped[cic_key] = val
    return mapped


class StreamCallbackWriter(OutputWriter):
    def __init__(self, preprocessor, buffer: FlowBuffer, writer: PredictionCSVWriter, feature_logger: Optional[FeatureLogger] = None):
        self.preprocessor = preprocessor
        self.buffer = buffer
        self.writer = writer
        self.feature_logger = feature_logger

    def write(self, data: dict) -> None:
        mapped = remap_cic_columns(data)
        if not mapped:
            return
        df = pd.DataFrame([mapped])
        try:
            X_proc = self.preprocessor.transform(df)
        except Exception:
            return
        feat = np.asarray(X_proc, dtype=np.float32).ravel()
        if self.feature_logger is not None:
            self.feature_logger.log(feat)
        result = self.buffer.add(feat)
        if result is not None:
            self.writer.write(result)


class StreamingCICInference:
    def __init__(
        self,
        interface: str,
        predictions_csv: str = "predictions.csv",
        stage1_path: Optional[str] = None,
        stage2_path: Optional[str] = None,
    ):
        self.interface = interface
        if stage1_path is not None or stage2_path is not None:
            self.cic = CICInference(
                stage1_path=stage1_path or CIC_STAGE1,
                stage2_path=stage2_path or CIC_STAGE2,
            )
        else:
            self.cic = CICInference()
        self.preprocessor = load_preprocessor(CICInference.PREPROCESSOR_PATH)
        self.writer = PredictionCSVWriter(predictions_csv)
        self.feature_logger = None

        n_features = 41
        self.buffer = FlowBuffer(
            window_size=CICInference.WINDOW_SIZE,
            on_prediction=self._predict_window,
            n_features=n_features,
        )

    def _predict_window(self, X_ae: np.ndarray, X_seq: np.ndarray) -> Dict:
        result = self.cic._predict_stages(X_ae, X_seq)[0]
        return result

    def run(self, log_features_path: Optional[str] = None):
        if log_features_path is not None:
            self.feature_logger = FeatureLogger(log_features_path, 41)
        cb_writer = StreamCallbackWriter(self.preprocessor, self.buffer, self.writer, feature_logger=self.feature_logger)
        session = FlowSession(output_mode="csv", output="/dev/null")
        session.output_writer = cb_writer

        _start_periodic_gc(session)

        scapy_conf.use_pcap = True
        sniffer = AsyncSniffer(
            iface=self.interface,
            filter="ip and (tcp or udp)",
            prn=session.process,
            store=False,
        )
        print(f"[CIC] Starting live capture on {self.interface}...")
        sniffer.start()
        try:
            sniffer.join()
        except KeyboardInterrupt:
            print("\n[CIC] Stopping...")
            sniffer.stop()
        finally:
            if hasattr(session, "_gc_stop"):
                session._gc_stop.set()
                session._gc_thread.join(timeout=2.0)
            sniffer.join()
            session.flush_flows()
            self.writer.close()
            if self.feature_logger is not None:
                self.feature_logger.close()
