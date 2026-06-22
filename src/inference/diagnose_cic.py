import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import tensorflow as tf

from keras.models import load_model
from sklearn.metrics import classification_report

from src.preprocessing.windowing.windowing import WindowGenerator
from src.utils.stage1_binary_scoring import apply_stage1_attack_score
from src.model.model import StopGradient
from src.grouping.definitions import (
    CIC_2GROUP_MAP, CIC_2GROUP_NAMES,
    build_group_mapping,
)

tf.random.set_seed(42)
import random
random.seed(42)
np.random.seed(42)

WINDOW_SIZE = 5
WINDOW_STEP = 1
DROP_COLUMNS = ["Label", "attack_label", "attack_type", "source_file"]
LABEL_COLUMN = "attack_type"
NORMAL_LABEL = "BENIGN"
CHUNK_SIZE = 30000

STAGE1_DIR = "models/classification/two_stage/cic/stage1"
STAGE1_MODEL_PATH = Path(STAGE1_DIR) / "best_model_binary.keras"
STAGE1_THRESHOLD_PATH = Path(STAGE1_DIR) / "threshold.json"
STAGE2_MODEL_PATH = "models/classification/two_stage/cic/stage2/twogroup/best_model_multiclass.keras"

# ── Load data ─────────────────────────────────────────────

print("loading data...")
test_df = pd.read_parquet("data/processed/CIC-IDS2017/splits/test/data.parquet")

label_encoder = joblib.load("models/preprocessing/multiclass/cic/label_encoder.pkl")
class_names = label_encoder.classes_.tolist()
NUM_CLASSES = len(label_encoder.classes_)
y_test_multi = label_encoder.transform(test_df[[LABEL_COLUMN]]).ravel()

# Build group mapping for reference
GROUP_NAMES = CIC_2GROUP_NAMES
original_to_group = build_group_mapping(
    label_encoder, CIC_2GROUP_MAP, GROUP_NAMES, NORMAL_LABEL
)
NUM_GROUPS = len(GROUP_NAMES)

# Binary labels
y_test_bin = (test_df[LABEL_COLUMN] != NORMAL_LABEL).astype(int).values

# Drop columns
X_test = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])
del test_df

# ── Preprocess ────────────────────────────────────────────

print("preprocessing...")
preprocessor = joblib.load("models/preprocessing/binary/cic/preprocessing.pkl")
X_test_proc = preprocessor.transform(X_test)
num_features = X_test_proc.shape[1]
del X_test

# ── Window ────────────────────────────────────────────────

print("windowing...")
window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=False)
X_test_ae, X_test_seq, y_test_w = window_builder.transform(X_test_proc, y_test_bin)
y_test_multi_w = window_builder._build_label_windows(y_test_multi)

y_raw = y_test_multi.copy()
del X_test_proc, y_test_bin, y_test_multi
print(f"    Windows: {X_test_seq.shape[0]}")

# ── Window purity analysis ────────────────────────────────

print("\n=== WINDOW PURITY ANALYSIS ===")
ws = WINDOW_SIZE
step = WINDOW_STEP

impure_by_class = Counter()
pure_count = 0
impure_count = 0
total_windows = 0

for start in range(0, len(y_raw) - ws + 1, step):
    window = y_raw[start:start + ws]
    unique = len(set(window))
    total_windows += 1
    if unique == 1:
        pure_count += 1
    else:
        impure_count += 1
        label = int(y_raw[start + ws - 1])
        impure_by_class[label] += 1

print(f"    Total windows: {total_windows}")
print(f"    Pure windows: {pure_count} ({pure_count/total_windows*100:.1f}%)")
print(f"    Impure (transition) windows: {impure_count} ({impure_count/total_windows*100:.1f}%)")
if impure_count > 0:
    print(f"\n    Impure windows by last-position label:")
    for label_idx, count in impure_by_class.most_common():
        cls_name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
        print(f"      {cls_name:<20} {count:>8} ({count/impure_count*100:.1f}%)")

# ── Stage 1 filter ───────────────────────────────────────

print("\n=== STAGE 1 FILTER ===")
stage1_model = load_model(STAGE1_MODEL_PATH, compile=False)
with open(STAGE1_THRESHOLD_PATH) as f:
    threshold_data = json.load(f)
THRESHOLD = threshold_data["threshold"]

def chunked_predict_and_score(model, ae, seq, dir, threshold_data, chunk_size=CHUNK_SIZE):
    n = len(ae)
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = {"ae_input": ae[start:end], "cnn_input": seq[start:end], "lstm_input": seq[start:end]}
        probs = model.predict(chunk, verbose=0)["classification"][:, 1]
        scores[start:end] = apply_stage1_attack_score(probs, dir, threshold_data)
    return scores

y_test_scores = chunked_predict_and_score(stage1_model, X_test_ae, X_test_seq, STAGE1_DIR, threshold_data)
attack_mask = y_test_scores > THRESHOLD
print(f"    Attack windows: {attack_mask.sum():,} / {len(attack_mask):,} ({attack_mask.sum()/len(attack_mask)*100:.1f}%)")

# ── Stage 2 inference ────────────────────────────────────

print("\n=== STAGE 2 INFERENCE ===")
stage2_model = load_model(STAGE2_MODEL_PATH, compile=False)

X_attack = {
    "ae_input": X_test_ae[attack_mask],
    "cnn_input": X_test_seq[attack_mask],
    "lstm_input": X_test_seq[attack_mask],
}
y_attack_orig = y_test_multi_w[attack_mask]  # original 8-class labels per window

# Each window already has a single label (last element of the window)
window_label_orig = y_attack_orig.copy()

# Predict groups
y_pred_probs = stage2_model.predict(X_attack, verbose=0)["classification"]
y_pred_group = np.argmax(y_pred_probs, axis=1)

# ── Per-class confusion ──────────────────────────────────

print("\n=== PER-CLASS CONFUSION (Original 8 classes vs Predicted group) ===")

# Manual rectangular confusion: original classes (rows) × predicted groups (cols)
confusion = np.zeros((NUM_CLASSES, NUM_GROUPS), dtype=np.int64)
for i in range(len(window_label_orig)):
    confusion[window_label_orig[i], y_pred_group[i]] += 1

print(f"\n{'Class':<25} {'Support':>8} {'→FloodBruteforce':>16} {'→Rare':>8} {'Misclass%':>10}")
print("-" * 70)
for orig_idx in range(len(class_names)):
    if orig_idx >= len(confusion):
        continue
    row = confusion[orig_idx]
    if len(row) < 2:
        continue
    total = row.sum()
    if total == 0:
        continue
    misclass = row[1]  # predicted as Rare (group 1)
    pct = misclass / total * 100
    print(f"{class_names[orig_idx]:<25} {total:>8} {row[0]:>16} {row[1]:>8} {pct:>9.1f}%")

# Also show within-attack-only (exclude BENIGN windows that passed Stage 1)
attack_only_mask = window_label_orig != 0  # 0 = BENIGN in label_encoder
print(f"\n--- Attack-only confusion (excluding BENIGN windows that passed Stage 1) ---")
print(f"    Attack windows with non-BENIGN labels: {attack_only_mask.sum():,}")

for orig_idx in range(len(class_names)):
    if orig_idx == 0:  # skip BENIGN
        continue
    if orig_idx >= len(confusion):
        continue
    row = confusion[orig_idx]
    if len(row) < 2:
        continue
    total = row.sum()
    if total == 0:
        continue
    misclass = row[1]
    pct = misclass / total * 100
    print(f"  {class_names[orig_idx]:<25} {total:>8} {row[0]:>16} {row[1]:>8} {pct:>9.1f}%")

# ── Purity within attack windows ─────────────────────────

print("\n=== PURITY WITHIN ATTACK WINDOWS ===")
attack_pure = 0
attack_impure = 0
for i in range(len(y_test_multi_w)):
    start = i * step
    window = y_raw[start:start + ws]
    if attack_mask[i]:
        if len(set(window)) == 1:
            attack_pure += 1
        else:
            attack_impure += 1
attack_total = attack_pure + attack_impure
if attack_total > 0:
    print(f"    Pure attack windows: {attack_pure} ({attack_pure/attack_total*100:.1f}%)")
    print(f"    Impure attack windows: {attack_impure} ({attack_impure/attack_total*100:.1f}%)")

# ── Full classification report per original class ────────

print("\n=== CLASSIFICATION REPORT (Group prediction per original class) ===")
report = classification_report(
    y_true=window_label_orig,
    y_pred=y_pred_group,
    target_names=class_names[:max(window_label_orig)+1],
    zero_division=0,
    output_dict=True,
)
report_df = pd.DataFrame(report).transpose()
print(report_df.to_string())

# Save
out_dir = Path("reports/metrics/cic/diagnostics/twogroup")
out_dir.mkdir(parents=True, exist_ok=True)
report_df.to_csv(out_dir / "per_class_confusion.csv")

# Detailed per-class confusion matrix
cm_df = pd.DataFrame(
    confusion,
    index=class_names[:len(confusion)],
    columns=GROUP_NAMES,
)
cm_df.to_csv(out_dir / "confusion_original_vs_group.csv")
print(f"\nResults saved to {out_dir}/")
