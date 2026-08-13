import csv
from datetime import datetime
import numpy as np
from collections import deque
from typing import Callable, Dict, List, Optional
from pathlib import Path


class FlowBuffer:
    def __init__(
        self,
        window_size: int,
        on_prediction: Callable[[np.ndarray, np.ndarray], Dict],
        n_features: int,
    ):
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size
        self.on_prediction = on_prediction
        self.n_features = n_features

    def add(self, feature_vector: np.ndarray) -> Optional[Dict]:
        self.buffer.append(feature_vector)
        if len(self.buffer) == self.window_size:
            arr = np.array(self.buffer, dtype=np.float32)
            X_ae = arr[-1:].reshape(1, self.n_features)
            X_seq = arr.reshape(1, self.window_size, self.n_features)
            return self.on_prediction(X_ae, X_seq)
        return None


class PredictionCSVWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", newline="")
        self.writer = None
        self.header_written = False

    def write(self, prediction: Dict):
        prediction["_timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        if not self.header_written:
            fieldnames = list(prediction.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
            self.writer.writeheader()
            self.header_written = True
        self.writer.writerow(prediction)
        self.file.flush()

    def close(self):
        self.file.close()


class FeatureLogger:
    def __init__(self, path: str, n_features: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        header = ["_timestamp"] + [f"f{i}" for i in range(n_features)]
        self.writer.writerow(header)
        self._count = 0

    def log(self, features: np.ndarray):
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        row = [ts] + [f"{v:.6f}" for v in features.ravel()]
        self.writer.writerow(row)
        self._count += 1
        if self._count % 100 == 0:
            self.file.flush()

    def close(self):
        self.file.close()
