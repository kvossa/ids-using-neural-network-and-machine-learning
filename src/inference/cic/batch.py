import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..base import BaseInference, load_model_safe
from src.grouping.definitions import CIC_BRUTERARE_NAMES
from src.utils.stage1_binary_scoring import apply_stage1_attack_score
from src.config import CIC_STAGE1, CIC_STAGE2, PREPROC_PATHS


class CICInference(BaseInference):
    PREPROCESSOR_PATH = PREPROC_PATHS["cic"]["binary_preprocessor"]
    LABEL_ENCODER_PATH = PREPROC_PATHS["cic"]["multiclass_encoder"]
    WINDOW_SIZE = 5
    DROP_COLUMNS = ["Label", "attack_label", "attack_type", "source_file"]
    NORMAL_LABEL = "BENIGN"

    GROUP_NAMES = CIC_BRUTERARE_NAMES

    def __init__(
        self,
        stage1_path: str = CIC_STAGE1,
        stage2_path: str = CIC_STAGE2,
    ):
        super().__init__()

        self.stage1_model = load_model_safe(stage1_path)
        self.stage2_model = load_model_safe(stage2_path)

        stage1_dir = Path(stage1_path).parent
        tpath = stage1_dir / "threshold.json"
        if tpath.exists():
            with open(tpath) as f:
                self._threshold_data = json.load(f)
            self.threshold = float(self._threshold_data["threshold"])
            self._stage1_dir = stage1_dir
            print(f"  Stage 1 threshold: {self.threshold:.4f} (calibration={self._threshold_data.get('calibration', 'none')})")
        else:
            self._threshold_data = None
            self.threshold = 0.5
            self._stage1_dir = None

    def predict_raw(
        self, df, return_probabilities: bool = False
    ) -> List[Dict]:
        X_proc = self._preprocess(df)
        X_ae, X_seq, _ = self._window(X_proc)
        return self._predict_stages(X_ae, X_seq, return_probabilities)

    def predict(
        self, X_ae: np.ndarray, X_seq: np.ndarray, return_probabilities: bool = False
    ) -> List[Dict]:
        return self._predict_stages(X_ae, X_seq, return_probabilities)

    def _predict_stages(
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

        stage1_output = self.stage1_model.predict(inputs, verbose=0)
        stage1_probs = stage1_output["classification"]
        stage1_recon = stage1_output["reconstruction"]
        raw_attack = stage1_probs[:, 1]

        if self._stage1_dir is not None and self._threshold_data is not None:
            stage1_scores = apply_stage1_attack_score(
                raw_attack, self._stage1_dir, self._threshold_data
            )
        else:
            stage1_scores = raw_attack

        stage1_pred = (stage1_scores > self.threshold).astype(int)
        stage1_confidence = np.max(stage1_probs, axis=1)

        attack_idx = np.where(stage1_pred == 1)[0]
        stage2_probs = None
        stage2_recon = None
        if len(attack_idx) > 0:
            batch_input = {
                "ae_input": X_ae[attack_idx],
                "cnn_input": X_seq[attack_idx],
                "lstm_input": X_seq[attack_idx],
            }
            stage2_output = self.stage2_model.predict(batch_input, verbose=0)
            stage2_probs = stage2_output["classification"]
            stage2_recon = stage2_output["reconstruction"]

        results = []
        pos = 0
        for i in range(len(X_ae)):
            if stage1_pred[i] == 0:
                mse = float(np.mean((X_ae[i] - stage1_recon[i]) ** 2))
                result = {
                    "prediction": self.NORMAL_LABEL,
                    "confidence": float(stage1_confidence[i]),
                    "group": None,
                    "stage1_result": self.NORMAL_LABEL,
                    "stage1_confidence": float(stage1_confidence[i]),
                    "reconstruction_mse": mse,
                }
            else:
                sp = stage2_probs[pos]
                sr = stage2_recon[pos]
                stage2_idx = int(np.argmax(sp))
                group_name = self.GROUP_NAMES[stage2_idx]
                mse = float(np.mean((X_ae[i] - sr) ** 2))

                result = {
                    "prediction": group_name,
                    "confidence": float(sp[stage2_idx]),
                    "group": group_name,
                    "stage1_result": "Attack",
                    "stage1_confidence": float(stage1_confidence[i]),
                    "reconstruction_mse": mse,
                }
                pos += 1

            if return_probabilities:
                result["stage1_probs"] = stage1_probs[i].tolist()
                if stage1_pred[i] == 1:
                    result["stage2_probs"] = sp.tolist()

            results.append(result)

        return results if is_batch else results[0]
