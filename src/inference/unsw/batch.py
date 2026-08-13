import numpy as np
from typing import Dict, List, Optional, Union

from ..base import BaseInference, load_model_safe
from src.grouping.definitions import (
    UNSW_CONFUSION_GROUP_MAP,
    UNSW_CONFUSION_GROUP_NAMES,
)
from src.config import PREPROC_PATHS, UNSW_MODEL


class UNSWInference(BaseInference):
    PREPROCESSOR_PATH = PREPROC_PATHS["unsw"]["binary_preprocessor"]
    LABEL_ENCODER_PATH = PREPROC_PATHS["unsw"]["multiclass_encoder"]
    WINDOW_SIZE = 10
    DROP_COLUMNS = ["attack_cat", "label", "id"]
    NORMAL_LABEL = "Normal"

    def __init__(
        self,
        model_path: str = UNSW_MODEL,
        merge_worms_into_exploits: bool = True,
    ):
        super().__init__()

        self.model = load_model_safe(model_path)
        self.merge_worms_into_exploits = merge_worms_into_exploits

        if self.merge_worms_into_exploits:
            self.group_map = dict(UNSW_CONFUSION_GROUP_MAP)
            self.group_map["Worms"] = "Exploits"
            self.group_names = [
                n for n in UNSW_CONFUSION_GROUP_NAMES if n != "Worms"
            ]
        else:
            self.group_map = UNSW_CONFUSION_GROUP_MAP
            self.group_names = UNSW_CONFUSION_GROUP_NAMES

    def predict_raw(
        self, df, return_probabilities: bool = False
    ) -> List[Dict]:
        X_proc = self._preprocess(df)
        X_ae, X_seq, _ = self._window(X_proc)
        return self._predict_model(X_ae, X_seq, return_probabilities)

    def predict(
        self, X_ae: np.ndarray, X_seq: np.ndarray, return_probabilities: bool = False
    ) -> List[Dict]:
        return self._predict_model(X_ae, X_seq, return_probabilities)

    def _predict_model(
        self, X_ae: np.ndarray, X_seq: np.ndarray, return_probabilities: bool = False
    ) -> List[Dict]:
        is_batch = len(X_ae.shape) == 2
        if not is_batch:
            X_ae = X_ae.reshape(1, -1)
            X_seq = X_seq.reshape(1, *X_seq.shape)

        inputs = {
            "ae_input": X_ae,
            "cnn_input": X_seq,
            "lstm_input": X_seq,
        }

        output = self.model.predict(inputs, verbose=0)
        probs = output["classification"]
        recon = output["reconstruction"]
        pred_indices = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        results = []
        for i in range(len(X_ae)):
            mse = float(np.mean((X_ae[i] - recon[i]) ** 2))
            group_name = self.group_names[pred_indices[i]]
            result = {
                "prediction": group_name,
                "confidence": float(confidence[i]),
                "group": group_name,
                "reconstruction_mse": mse,
            }
            if return_probabilities:
                result["probs"] = probs[i].tolist()
            results.append(result)

        return results if is_batch else results[0]
