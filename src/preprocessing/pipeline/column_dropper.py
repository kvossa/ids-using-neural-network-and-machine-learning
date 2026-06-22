import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


CIC_DROP = [
    "Bwd Avg Bulk Rate",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bytes/Bulk",
    "Fwd Avg Bulk Rate",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bytes/Bulk",
    "Bwd URG Flags",
    "Bwd PSH Flags",
    "CWE Flag Count",
    "Fwd URG Flags",
    "ECE Flag Count",
    "FIN Flag Count",
    "Avg Bwd Segment Size",
    "Avg Packet Size",
    "Bwd Packet Length Max",
    "Bwd Packet Length Std",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Flow IAT Max",
    "Fwd IAT Max",
    "Fwd IAT Std",
    "Idle Max",
    "Idle Min",
    "Bwd Packets Length Total",
    "Subflow Bwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Fwd Packets",
    "Fwd IAT Total",
    "Fwd Packets Length Total",
    "Subflow Fwd Bytes",
    "Fwd Packet Length Std",
    "Avg Fwd Segment Size",
    "Fwd IAT Mean",
    "Bwd IAT Min",
    "Active Min",
]

UNSW_DROP = [
    "sloss",
    "dloss",
    "ackdat",
    "is_sm_ips_ports",
    "dwin",
]


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, dataset="CIC"):
        self.dataset = dataset.upper()
        self.columns_to_drop_ = None

    def fit(self, X, y=None):
        if self.dataset == "CIC":
            self.columns_to_drop_ = [c for c in CIC_DROP if c in X.columns]
        elif self.dataset == "UNSW":
            self.columns_to_drop_ = [c for c in UNSW_DROP if c in X.columns]
        else:
            self.columns_to_drop_ = []
        return self

    def transform(self, X):
        X_dropped = X.drop(columns=self.columns_to_drop_, errors="ignore")
        return X_dropped
