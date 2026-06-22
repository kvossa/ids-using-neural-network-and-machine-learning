import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import tensorflow as tf

from keras.models import load_model
from sklearn.metrics import classification_report

from src.preprocessing.windowing.windowing import WindowGenerator
from src.model.model import StopGradient
from src.grouping.definitions import (
    UNSW_CONFUSION_GROUP_MAP, UNSW_CONFUSION_GROUP_NAMES,
    build_group_mapping,
)

tf.random.set_seed(42)
import random
random.seed(42)
np.random.seed(42)

WINDOW_SIZE = 10
WINDOW_STEP = 1
DROP_COLUMNS = ["attack_cat", "label", "id"]
LABEL_COLUMN = "attack_cat"
NORMAL_LABEL = "Normal"
MERGE_WORMS_INTO_EXPLOITS = True

MODEL_PATH = "models/classification/single_stage/unsw/confusion_groups/baseline/best_model_multiclass.keras"

# ── Load data ─────────────────────────────────────────────

print("loading data...")
test_df = pd.read_csv("data/processed/UNSW-NB15/splits/test.csv")

label_encoder = joblib.load("models/preprocessing/multiclass/unsw/label_encoder.pkl")
y_test_multi = label_encoder.transform(test_df[[LABEL_COLUMN]])

NUM_CLASSES = len(label_encoder.classes_)
print(f"Classes ({NUM_CLASSES}): {label_encoder.classes_.tolist()}")

# Build group mapping
group_map = dict(UNSW_CONFUSION_GROUP_MAP)
if MERGE_WORMS_INTO_EXPLOITS:
    group_map["Worms"] = "Exploits"
    GROUP_NAMES = [g for g in UNSW_CONFUSION_GROUP_NAMES if g != "Worms"]
else:
    GROUP_NAMES = UNSW_CONFUSION_GROUP_NAMES

original_to_group = build_group_mapping(
    label_encoder, group_map, GROUP_NAMES, normal_label=None
)
NUM_GROUPS = len(GROUP_NAMES)
print(f"Groups ({NUM_GROUPS}): {GROUP_NAMES}")

# Binary labels
y_test_bin = (test_df[LABEL_COLUMN] != NORMAL_LABEL).astype(int).values

# Drop columns
X_test = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])
del test_df

# ── Preprocess ────────────────────────────────────────────

print("preprocessing...")
preprocessor = joblib.load("models/preprocessing/binary/unsw/preprocessing.pkl")
X_test_proc = preprocessor.transform(X_test)
num_features = X_test_proc.shape[1]
del X_test

# ── Window ────────────────────────────────────────────────

print("windowing...")
window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=False)
X_test_ae, X_test_seq, _ = window_builder.transform(X_test_proc, np.zeros(len(X_test_proc)))
y_test_multi_w = window_builder._build_label_windows(y_test_multi)
y_raw = y_test_multi.copy()

del X_test_proc
print(f"    Windows: {X_test_seq.shape[0]}")

# ── Window purity analysis ────────────────────────────────

print("\n=== WINDOW PURITY ANALYSIS ===")
class_names = label_encoder.classes_.tolist()
ws = WINDOW_SIZE
step = WINDOW_STEP

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

print(f"    Total windows: {total_windows}")
print(f"    Pure windows: {pure_count} ({pure_count/total_windows*100:.1f}%)")
print(f"    Impure (transition) windows: {impure_count} ({impure_count/total_windows*100:.1f}%)")

# ── Model inference ──────────────────────────────────────

print("\n=== MODEL INFERENCE ===")
model = load_model(MODEL_PATH, compile=False)

X_all = {
    "ae_input": X_test_ae,
    "cnn_input": X_test_seq,
    "lstm_input": X_test_seq,
}

y_pred_probs = model.predict(X_all, verbose=0)["classification"]
y_pred_group = np.argmax(y_pred_probs, axis=1)

# Each window already has a single label (last element of the window)
window_label_orig = y_test_multi_w.copy()

# Map original labels to groups (same mapping used during training)
y_true_group = original_to_group[window_label_orig]

# ── Per-class confusion ──────────────────────────────────

print("\n=== PER-CLASS CONFUSION (Original 10 classes vs Predicted group) ===")

# Manual rectangular confusion: original classes (rows) × predicted groups (cols)
confusion = np.zeros((NUM_CLASSES, NUM_GROUPS), dtype=np.int64)
for i in range(len(window_label_orig)):
    confusion[window_label_orig[i], y_pred_group[i]] += 1

print(f"\n{'Class':<20} {'Support':>8} → Groups", end="")
for g in range(NUM_GROUPS):
    print(f"{GROUP_NAMES[g]:>20}", end="")
print(f" {'PredomGroup':>15}")

print("-" * (20 + 8 + 20 * NUM_GROUPS + 15))
for orig_idx in range(NUM_CLASSES):
    row = confusion[orig_idx]
    total = row.sum()
    if total == 0:
        continue
    predominant = np.argmax(row)
    line = f"{class_names[orig_idx]:<20} {total:>8}"
    for g in range(NUM_GROUPS):
        line += f" {row[g]:>20}"
    line += f" {GROUP_NAMES[predominant]:>15}"
    print(line)

# ── Within-Medium breakdown ──────────────────────────────

print("\n=== MEDIUM GROUP BREAKDOWN ===")
medium_group_idx = GROUP_NAMES.index("Medium")
medium_mask = y_true_group == medium_group_idx
print(f"    Windows where true group is Medium: {medium_mask.sum():,}")

medium_orig_labels = window_label_orig[medium_mask]
medium_pred_groups = y_pred_group[medium_mask]

print(f"\n{'Original Class':<20} {'Count':>8} {'→Correct(Medium)':>18} {'→Other':>8} {'Misclass%':>10}")
print("-" * (20 + 8 + 18 + 8 + 10))
for orig_idx in range(NUM_CLASSES):
    mask = medium_orig_labels == orig_idx
    count = mask.sum()
    if count == 0:
        continue
    correct = (medium_pred_groups[mask] == medium_group_idx).sum()
    misclass_pct = (count - correct) / count * 100
    print(f"{class_names[orig_idx]:<20} {count:>8} {correct:>18} {count - correct:>8} {misclass_pct:>9.1f}%")

# ── Full classification report ───────────────────────────

print("\n=== CLASSIFICATION REPORT ===")
report = classification_report(
    y_true=y_true_group,
    y_pred=y_pred_group,
    target_names=GROUP_NAMES,
    zero_division=0,
    output_dict=True,
)
report_df = pd.DataFrame(report).transpose()
print(report_df.to_string())

# Save
out_dir = Path("reports/metrics/unsw/diagnostics")
out_dir.mkdir(parents=True, exist_ok=True)
report_df.to_csv(out_dir / "classification_report.csv")

# Per-class confusion
cm_df = pd.DataFrame(
    confusion,
    index=class_names,
    columns=GROUP_NAMES,
)
cm_df.to_csv(out_dir / "confusion_original_vs_group.csv")

# Medium breakdown
medium_breakdown = []
for orig_idx in range(NUM_CLASSES):
    mask = medium_orig_labels == orig_idx
    count = mask.sum()
    if count == 0:
        continue
    correct = (medium_pred_groups[mask] == medium_group_idx).sum()
    medium_breakdown.append({
        "original_class": class_names[orig_idx],
        "count": count,
        "correct_medium": correct,
        "misclassified": count - correct,
        "misclass_pct": (count - correct) / count * 100,
    })

pd.DataFrame(medium_breakdown).to_csv(out_dir / "medium_breakdown.csv", index=False)

print(f"\nResults saved to {out_dir}/")
