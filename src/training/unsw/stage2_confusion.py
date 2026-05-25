#!/usr/bin/env python
"""
Stage 2 Diagnostic - Generate Visualizations from earlier results
Uses existing best model to produce confusion matrices
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import src.model.model

STAGE2_DIR = Path("models/classification/two_stage/unsw/stage2")
OUTPUT_DIR = Path("reports/figures/unsw/two_stage/stage2")
TEST_DATA = Path("data/processed/UNSW-NB15/splits/test.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GENERATING VISUAL CONFUSION MATRIX")
print("=" * 60)

# Load preprocessor and encoder
preprocessor = joblib.load("models/preprocessing/binary/unsw/preprocessing.pkl")
label_encoder = joblib.load("models/preprocessing/multiclass/unsw/label_encoder.pkl")
class_names = label_encoder.classes_.tolist()
NUM_CLASSES = len(class_names)

# Load data
print("\nLoading test data...")
test_df = pd.read_csv(TEST_DATA)
NORMAL_LABEL = "Normal"
test_df = test_df[test_df['attack_cat'] != NORMAL_LABEL].reset_index(drop=True)

X_test = test_df.drop(columns=['attack_cat', 'label', 'id'])
y_test = label_encoder.transform(test_df[['attack_cat']])
X_test_proc = preprocessor.transform(X_test)

print(f"  Test samples: {len(X_test_proc)}")
print(f"  Classes: {class_names}")

# Load ensemble model and predict
print("\nLoading ensemble...")
ensemble_models = sorted(STAGE2_DIR.glob("ensemble_model_*.keras"))
if len(ensemble_models) >= 3:
    models = [load_model(str(m)) for m in ensemble_models]
    print(f"  Using {len(models)} ensemble models")
    
    # Window and predict
    from src.preprocessing.windowing import WindowGenerator
    window_builder = WindowGenerator(window_size=10, step=1, pure_windows_only=False)
    y_binary = np.ones(len(X_test_proc))
    X_test_ae, X_test_seq, _ = window_builder.transform(X_test_proc, y_binary)
    
    # Get aligned labels
    y_full = label_encoder.transform(test_df[['attack_cat']])
    y_windows = window_builder._build_label_windows(y_full.ravel())
    
    # Predict with ensemble
    probs_list = []
    for m in models:
        probs = m.predict(
            {"ae_input": X_test_ae, "cnn_input": X_test_seq, "lstm_input": X_test_seq},
            verbose=0
        )["classification"]
        probs_list.append(probs)
    
    y_pred_probs = np.mean(probs_list, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Ensure alignment
    min_len = min(len(y_windows), len(y_pred))
    y_true = y_windows[:min_len]
    y_pred = y_pred[:min_len]
else:
    print("ERROR: No ensemble models found!")
    exit(1)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# === PLOT 1: RAW COUNTS ===
print("\nGenerating plots...")
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title('Stage 2 Confusion Matrix (Counts)', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_counts.png", dpi=150)
print(f"  Saved: confusion_matrix_counts.png")
plt.close()

# === PLOT 2: NORMALIZED ===
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            vmin=0, vmax=1)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title('Stage 2 Confusion Matrix (Row-Normalized %)', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_normalized.png", dpi=150)
print(f"  Saved: confusion_matrix_normalized.png")
plt.close()

# === PLOT 3: PER-CLASS ACCURACY ===
acc_per_class = []
for i in range(NUM_CLASSES):
    if cm[i].sum() > 0:
        acc_per_class.append(cm[i,i] / cm[i].sum() * 100)
    else:
        acc_per_class.append(0)

sorted_idx = np.argsort(acc_per_class)
colors = ['#d62728' if acc < 30 else '#ff7f0e' if acc < 50 else '#2ca02c' for acc in acc_per_class]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh([class_names[i] for i in sorted_idx], [acc_per_class[i] for i in sorted_idx],
       color=[colors[i] for i in sorted_idx])
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Class Accuracy (Sorted)', fontsize=14)
ax.set_xlim(0, 100)

for bar, acc in zip(bars, [acc_per_class[i] for i in sorted_idx]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{acc:.1f}%', va='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "per_class_accuracy.png", dpi=150)
print(f"  Saved: per_class_accuracy.png")
plt.close()

# === PLOT 4: F1 SCORES ===
f1_per_class = []
report = classification_report(y_true, y_pred, labels=range(NUM_CLASSES), 
                           target_names=class_names, output_dict=True, zero_division=0)
for cn in class_names:
    f1_per_class.append(report[cn]['f1-score'] * 100)

sorted_idx = np.argsort(f1_per_class)
colors = ['#d62728' if f1 < 30 else '#ff7f0e' if f1 < 50 else '#2ca02c' for f1 in f1_per_class]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh([class_names[i] for i in sorted_idx], [f1_per_class[i] for i in sorted_idx],
       color=[colors[i] for i in sorted_idx])
ax.set_xlabel('F1 Score (%)', fontsize=12)
ax.set_title('Per-Class F1 Score (Sorted)', fontsize=14)
ax.set_xlim(0, 100)

for bar, f1 in zip(bars, [f1_per_class[i] for i in sorted_idx]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{f1:.1f}%', va='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "per_class_f1.png", dpi=150)
print(f"  Saved: per_class_f1.png")
plt.close()

# === SUMMARY ===
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"\nMacro F1: {macro_f1:.4f}")

# Top misclassifications
print("\nTop misclassifications (True -> Predicted):")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i,j] > 0:
            pct = cm[i,j] / cm[i].sum() * 100
            if pct > 3:  # Only show >3%
                print(f"  {class_names[i]:15} -> {class_names[j]:15}: {cm[i,j]:4d} ({pct:5.1f}%)")

# Worst classes
print("\nWorst 3 classes (by accuracy):")
for i in sorted_idx[:3]:
    print(f"  {class_names[i]}: {acc_per_class[i]:.1f}%")

print("\n" + "=" * 60)
print("COMPLETE - Check reports/figures/unsw/two_stage/stage2/")
print("=" * 60)