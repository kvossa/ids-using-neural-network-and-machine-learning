#!/usr/bin/env python
"""
Stage 2 Diagnostic - Simple version
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score
import src.model.model

STAGE2_DIR = Path("models/classification/two_stage/unsw/stage2")
TEST_DATA = Path("data/processed/UNSW-NB15/splits/test.csv")

print("=" * 60)
print("STAGE 2 DIAGNOSTIC")
print("=" * 60)

# Load data
test_df = pd.read_csv(TEST_DATA)
NORMAL_LABEL = "Normal"
test_df = test_df[test_df['attack_cat'] != NORMAL_LABEL].reset_index(drop=True)

label_encoder = joblib.load("models/preprocessing/multiclass/unsw/label_encoder.pkl")
preprocessor = joblib.load("models/preprocessing/binary/unsw/preprocessing.pkl")

X_test = test_df.drop(columns=['attack_cat', 'label', 'id'])
y_test = label_encoder.transform(test_df[['attack_cat']])
X_test_proc = preprocessor.transform(X_test)

# Get AE features (use mean of window)
X_test_ae = X_test_proc.mean(axis=1)  # Simple: average of window
X_test_seq = np.expand_dims(X_test_proc, axis=1)  # Add sequence dim

print(f"\n    Test attacks: {len(X_test_ae)}")

# Load model
ensemble_models = sorted(STAGE2_DIR.glob("ensemble_model_*.keras"))
if ensemble_models:
    print(f"    Using {len(ensemble_models)} ensemble models")
    models = [load_model(str(m)) for m in ensemble_models]
    probs_list = [m.predict({"ae_input": X_test_ae, "cnn_input": X_test_seq, "lstm_input": X_test_seq}, verbose=0)["classification"] for m in models]
    y_pred_probs = np.mean(probs_list, axis=0)
else:
    model = load_model(str(STAGE2_DIR / "best_model_multiclass.keras"))
    y_pred_probs = model.predict({"ae_input": X_test_ae, "cnn_input": X_test_seq, "lstm_input": X_test_seq}, verbose=0)["classification"]

y_pred = np.argmax(y_pred_probs, axis=1)

# === METRICS ===
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n  Macro F1: {f1:.4f}")

# Per class
class_names = label_encoder.classes_.tolist()
print("\n  Per-class accuracy:")
for i, cn in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).sum() / mask.sum() * 100
        print(f"    {cn:15s}: {acc:.1f}%")

# Confusion pairs
print("\n" + "=" * 60)
print("TOP CONFUSIONS")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
errors = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i,j] > 0:
            errors.append((class_names[i], class_names[j], cm[i,j]))

errors.sort(key=lambda x: x[2], reverse=True)
print("\nMost common misclassifications:")
for true_c, pred_c, count in errors[:10]:
    pct = count / cm[true_c == class_names.index(true_c)].sum() * 100 if cm[true_c == class_names.index(true_c)].sum() > 0 else 0
    print(f"  {true_c} -> {pred_c}: {count}")

print("\n" + "=" * 60)