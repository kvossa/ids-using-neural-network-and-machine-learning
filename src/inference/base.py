import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Optional

import src.model.model  # registers StopGradient

from src.preprocessing.windowing.windowing import WindowGenerator


def load_model_safe(path: str):
    return tf.keras.models.load_model(path, compile=False)


def load_preprocessor(path: str):
    return joblib.load(path)


def load_label_encoder(path: str):
    return joblib.load(path)


class BaseInference:
    PREPROCESSOR_PATH: str
    LABEL_ENCODER_PATH: str
    WINDOW_SIZE: int
    DROP_COLUMNS: List[str]

    def __init__(self):
        self.preprocessor = load_preprocessor(self.PREPROCESSOR_PATH)
        self.label_encoder = load_label_encoder(self.LABEL_ENCODER_PATH)
        self.window_gen = WindowGenerator(
            window_size=self.WINDOW_SIZE, step=1, pure_windows_only=False
        )

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=[c for c in self.DROP_COLUMNS if c in df.columns])
        return self.preprocessor.transform(X)

    def _window(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dummy_y = np.zeros(len(X), dtype=np.int32)
        return self.window_gen.transform(X, dummy_y)

    def preprocess_and_window(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_proc = self._preprocess(df)
        X_ae, X_seq, _ = self._window(X_proc)
        return X_ae, X_seq
