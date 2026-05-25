import numpy as np
import tensorflow as tf
import joblib
import json
from pathlib import Path
from typing import Dict, Union, Optional

from keras.models import load_model

from src.utils.stage1_binary_scoring import apply_stage1_attack_score

class TwoStageDetection:
    def __init__(self, stage1_path:str, stage2_path:str, stage1_threshold:float=0.3, 
                label_encoder_path:str='models/preprocessing/multiclass/unsw/label_encoder.pkl'):
        stage1_dir = Path(stage1_path).parent
        tpath = stage1_dir / "threshold.json"
        if tpath.exists():
            with open(tpath) as f:
                self._stage1_threshold_data = json.load(f)
            self.threshold = float(self._stage1_threshold_data["threshold"])
            self._stage1_dir = stage1_dir
            print(f'    Stage 1 threshold from {tpath}: {self.threshold} (calibration={self._stage1_threshold_data.get("calibration", "none")})')
        else:
            self._stage1_threshold_data = None
            self.threshold = float(stage1_threshold)
            self._stage1_dir = None

        print(f'loading stage 1 model from {stage1_path}...')
        self.stage1_model = load_model(stage1_path, compile=False)

        print(f'loading stage 2 model from {stage2_path}...')
        self.stage2_model = load_model(stage2_path, compile=False)

        print(f'loading label encoder model from {label_encoder_path}...')
        self.label_encoder = joblib.load(label_encoder_path)

        self.attack_classes = [c for c in self.label_encoder.classes_ if c != "Normal"]

    def predict(self, X_ae:np.ndarray, X_seq:np.ndarray, return_probabilities:bool=False)->Union[Dict, list]:
        is_batch = len(X_ae.shape) == 2

        if not is_batch:
            X_ae = X_ae.reshape(1, -1)
            X_seq = X_seq.reshape(1, *X_seq.shape)

        inputs = {
            "ae_input": X_ae,
            "cnn_input": X_seq,
            "lstm_input": X_seq
        }

        stage1_probs = self.stage1_model.predict(inputs, verbose=0)["classification"]
        raw_attack = stage1_probs[:, 1]
        if self._stage1_threshold_data is not None:
            stage1_scores = apply_stage1_attack_score(
                raw_attack, self._stage1_dir, self._stage1_threshold_data
            )
        else:
            stage1_scores = raw_attack
        stage1_pred = (stage1_scores > self.threshold).astype(int)
        stage1_confidence = np.max(stage1_probs, axis=1)

        results = [] 

        for i in range(len(X_ae)):
            if stage1_pred[i] == 0:
                result = {
                    "stage1": "Normal",
                    "stage1_confidence": float(stage1_confidence[i]),
                    "stage2": None,
                    "stage2_confidence": None
                }
            else:
                sample_input = {
                    "ae_input": X_ae[i:i+1],
                    "cnn_input": X_seq[i:i+1],
                    "lstm_input": X_seq[i:i+1],
                }
                stage2_probs = self.stage2_model.predict(sample_input, verbose=0)["classification"]
                stage2_pred_idx = np.argmax(stage2_probs)
                stage2_pred = self.label_encoder.inverse_transform([stage2_pred_idx])[0]

                result = {
                    "stage1": "Attack",
                    "stage1_confidence": float(stage1_confidence[i]),
                    "stage2": stage2_pred,
                    "stage2_confidence": float(stage2_probs[0, stage2_pred_idx])
                }

            if return_probabilities:
                result["stage_probs"] = stage1_probs[i].tolist()
                if stage1_pred == 1:
                    result["stage2_probs"] = stage2_probs[0].tolist()

            results.append(result)

        return results if is_batch else results[0]

    def predict_batch(self, X_ae:np.ndarray, X_seq:np.ndarray)->list:
        return self.predict(X_ae, X_seq)

def main():
    print("\nTwo Stage Detection Demonstration")

    detector = TwoStageDetection(
        stage1_path='models/classification/two_stage/unsw/stage1/best_model_binary.keras',
        stage2_path='models/classification/two_stage/unsw/stage2/best_model_multiclass.keras',
        label_encoder_path='preprocessing/multiclass/unsw/label_encoder.pkl'
    )

    num_features = 42
    window_size = 10

    X_ae = np.random.randn(5, num_features).astype(np.float32)
    X_seq = np.random.randn(5, window_size, num_features).astype(np.float32)

    results = detector.predict(X_ae=X_ae, X_seq=X_seq, return_probabilities=True)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()