import numpy as np
import pandas as pd
from typing import Optional, Tuple

class WindowGenerator:
    def __init__(self, window_size:int=20, step:int=1, min_flow_lenght:int=2, pure_windows_only:bool=False):
        self.window_size = window_size
        self.step = step
        self.min_flow_lenght = min_flow_lenght
        self.pure_windows_only = pure_windows_only

    def fit(self, X, y=None):
        return self

    def transform(self, X:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = X.values if hasattr(X, "values") else np.array(X)
        y = y.values if hasattr(y, "values") else np.array(y)
        y = y.ravel()
        return self._build_global_windows(X, y)

    def _build_global_windows(self, X:np.ndarray, y:np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        ws = self.window_size
        step = self.step
        
        if n<ws:
            raise ValueError(f"La ventana es más grande que el número de muestras")

        n_windows = ((n - ws) // step) + 1
        n_features = X.shape[1]

        # Preallocate to avoid Python-list overhead and double-memory copies.
        X_seq = np.empty((n_windows, ws, n_features), dtype=np.float32)
        X_ae = np.empty((n_windows, n_features), dtype=np.float32)
        y_out = np.empty((n_windows,), dtype=y.dtype)

        windows_skipped = 0
        kept = 0

        for start in range(0, n - ws + 1, step):
            end = start + ws
            y_window = y[start:end]

            if self.pure_windows_only and not self._check_window_purity(y_window=y_window):
                windows_skipped+=1
                continue

            X_seq[kept] = X[start:end]
            X_ae[kept] = X[end-1]
            y_out[kept] = y[end-1]
            kept += 1

        if windows_skipped:
            pct = windows_skipped / ((n - ws) // step + 1) * 100
            print(f"[WindowBuilder] Ventanas impuras descartadas: {windows_skipped:,} ({pct:.1f}%)")
            if pct > 40:
                print(f"CUIDADO Más del 40% descartado")

        return (
            X_ae[:kept],
            X_seq[:kept],
            y_out[:kept],
        )

        # X_seq = np.empty((n_windows, ws, X.shape[1]), dtype=np.float32)
        # X_ae = np.empty((n_windows, ws, X.shape[1]), dtype=np.float32)
        # y_out = np.empty(n_windows, dtype=y.dtype)

        # for i, start in enumerate(indices):
        #     end = start + ws
        #     X_seq[i] = X[start:end]
        #     X_ae[i] = X[end-1]
        #     y_out = y[end-1]
        
        # return X_ae, X_seq, y_out
    
    def _check_window_purity(self, y_window: np.ndarray)->bool:
        return len(np.unique(y_window)) == 1

    def _build_label_windows(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        n = len(y)
        ws = self.window_size
        step = self.step
        if n < ws:
            raise ValueError(f"La ventana es mas grande que el numero de muestras")
        indices = range(0, n - ws + 1, step)
        return np.array([y[start + ws - 1] for start in indices])
