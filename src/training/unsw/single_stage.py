import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
tf.random.set_seed(42)

from keras.metrics import AUC, Precision, Recall, F1Score
from keras.utils import to_categorical
from keras.losses import CategoricalFocalCrossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import ADASYN

from src.model.model import IDSModelFactory
from src.preprocessing.windowing.windowing import WindowGenerator
from src.utils.batch_balancer import create_hybrid_mix_dataset, create_class_balanced_dataset
from src.utils.train_stopper import F1EarlyStopping
from src.grouping.definitions import (
    UNSW_CONFUSION_GROUP_MAP, UNSW_CONFUSION_GROUP_NAMES,
    UNSW_CONFUSION_OVERSAMPLE_RATES, UNSW_CONFUSION_CLASS_ALPHA,
    build_group_mapping,
)
from src.config import DATA_PATHS, PREPROC_PATHS, REPORT_PATHS, UNSW_CONFUSION_GROUPS_DIR

import random
random.seed(42)
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────
# Grouping
MERGE_WORMS_INTO_EXPLOITS = True    # False = 7 groups (Worms standalone)
                                    # True  = 6 groups (Worms→Exploits)

# Resampling before windowing
RESAMPLE_TARGET_RATIO = 0.50        # target = min(count*8, majority * this)

# Post-windowing oversampling (hybrid mix)
OVERSAMPLE_RATES = UNSW_CONFUSION_OVERSAMPLE_RATES
JITTER_STD = 0.05
ORIGINAL_MIX_RATIO = 0.7

# Architecture
HEAD_DEPTH = "attention"            # "standard", "attention"
ENSEMBLE_MODELS = 1

# Training
WINDOW_SIZE = 10
WINDOW_STEP = 1
CHUNK_SIZE = 50000
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 10
LEARNING_RATE = 5e-5
HEAD_LR = 1e-3
FOCAL_GAMMA = 3.0

# Model architecture config (toggle between baseline and experiment)
MODEL_CONFIG = {
    "conv_l2": 0.001,
    "use_first_bn": False,
    "bn_momentum": 0.9,
    "rnn_type": "lstm",
    "rnn_units": 64,
    "rnn_layers": 2,
}
MODEL_TAG = "baseline"

CLASS_ALPHA = UNSW_CONFUSION_CLASS_ALPHA

DROP_COLUMNS = ["attack_cat", "label", "id"]
LABEL_COLUMN = "attack_cat"
NORMAL_LABEL = "Normal"

REPORTS_PATH = Path(REPORT_PATHS["unsw_confusion_groups"]) / MODEL_TAG
MODELS_PATH = Path(UNSW_CONFUSION_GROUPS_DIR) / MODEL_TAG

for p in [REPORTS_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"    UNSW Single-Stage Multiclass Classification")
print(f"\n{'='*60}\n")

print("loading data...")
train_df = pd.read_csv(DATA_PATHS["unsw"]["train"])
test_df = pd.read_csv(DATA_PATHS["unsw"]["test"])
val_df = pd.read_csv(DATA_PATHS["unsw"]["val"])

y_train_raw = train_df[LABEL_COLUMN]
y_test_raw = test_df[LABEL_COLUMN]
y_val_raw = val_df[LABEL_COLUMN]

X_train = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])
X_val = val_df.drop(columns=[c for c in DROP_COLUMNS if c in val_df.columns])

label_encoder = joblib.load(PREPROC_PATHS["unsw"]["multiclass_encoder"])

y_train_multi = label_encoder.transform(train_df[[LABEL_COLUMN]])
y_test_multi = label_encoder.transform(test_df[[LABEL_COLUMN]])
y_val_multi = label_encoder.transform(val_df[[LABEL_COLUMN]])

NUM_CLASSES = len(label_encoder.classes_)
print(f"Original classes: {NUM_CLASSES}")
print(f"Classes: {label_encoder.classes_.tolist()}")

# Build confusion-based group mapping
if MERGE_WORMS_INTO_EXPLOITS:
    group_map = dict(UNSW_CONFUSION_GROUP_MAP)
    group_map["Worms"] = "Exploits"
    GROUP_NAMES = [n for n in UNSW_CONFUSION_GROUP_NAMES if n != "Worms"]
    OVERSAMPLE_RATES = {k: v for k, v in UNSW_CONFUSION_OVERSAMPLE_RATES.items() if k != 6}
    CLASS_ALPHA = {k: v for k, v in UNSW_CONFUSION_CLASS_ALPHA.items() if k != 6}
else:
    group_map = UNSW_CONFUSION_GROUP_MAP
    GROUP_NAMES = list(UNSW_CONFUSION_GROUP_NAMES)

original_to_group = build_group_mapping(
    label_encoder, group_map, GROUP_NAMES, normal_label=None
)
NUM_GROUPS = len(GROUP_NAMES)
print(f"Groups ({NUM_GROUPS}): {GROUP_NAMES}")

print("Preprocessing...")
preprocessor = joblib.load(PREPROC_PATHS["unsw"]["binary_preprocessor"])

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_val_proc = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]

# ── Pre-windowing resampling (ADASYN) ──
print("\n applying ADASYN before windowing...")
y_train_group = original_to_group[y_train_multi]

group_counts = np.bincount(y_train_group, minlength=NUM_GROUPS)
majority_count = group_counts.max()
print(f"    Pre-resample distribution:")
for i in range(NUM_GROUPS):
    print(f"      {GROUP_NAMES[i]:<20s} {group_counts[i]:>6,}")

# Target: boost minority groups to RESAMPLE_TARGET_RATIO of majority (capped)
resample_targets = {}
for g in range(NUM_GROUPS):
    if group_counts[g] < majority_count * 0.3:
        target = min(int(group_counts[g] * 8), int(majority_count * RESAMPLE_TARGET_RATIO))
        resample_targets[g] = target

if resample_targets:
    k = min(5, min(group_counts[g] for g in resample_targets) - 1)
    k = max(1, k)

    sampler = ADASYN(sampling_strategy=resample_targets, n_neighbors=k, random_state=42)
    X_train_proc, y_train_group = sampler.fit_resample(X_train_proc, y_train_group)
    print(f"\n    ADASYN augmented: {len(X_train_proc):,} rows (from {group_counts.sum():,})")
    for i in range(NUM_GROUPS):
        cnt = int((y_train_group == i).sum())
        print(f"      {GROUP_NAMES[i]:<20s} {cnt:>6,}")
else:
    print("    No groups meet threshold, skipping resampling.")

y_train_binary = (y_train_group != 0).astype(int)  # 0 = Normal

# ── Chunked Windowing (reduces peak memory) ───────────────────
def chunked_window_transform(X, y, chunk_size=CHUNK_SIZE):
    n = len(X)
    ws = WINDOW_SIZE
    step = WINDOW_STEP
    n_windows = ((n - ws) // step) + 1
    n_features = X.shape[1]

    X_ae = np.empty((n_windows, n_features), dtype=np.float32)
    X_seq = np.empty((n_windows, ws, n_features), dtype=np.float32)

    num_chunks = (n_windows + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        w_start = i * chunk_size
        w_end = min((i + 1) * chunk_size, n_windows)
        src_start = w_start
        src_end = min(w_end + ws - 1, n)

        chunk_X = X[src_start:src_end]
        chunk_y = y[src_start:src_end]
        chunk_ae, chunk_seq, _ = window_builder.transform(chunk_X, chunk_y)

        n_actual = w_end - w_start
        X_ae[w_start:w_end] = chunk_ae[:n_actual]
        X_seq[w_start:w_end] = chunk_seq[:n_actual]

        del chunk_X, chunk_ae, chunk_seq, chunk_y

    return X_ae, X_seq

print("\n windowing...")
window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=False)

X_train_ae, X_train_seq = chunked_window_transform(X_train_proc, y_train_binary)
X_test_ae, X_test_seq, _ = window_builder.transform(X_test_proc, (y_test_raw != NORMAL_LABEL).astype(int))
X_val_ae, X_val_seq, _ = window_builder.transform(X_val_proc, (y_val_raw != NORMAL_LABEL).astype(int))

y_train_data = window_builder._build_label_windows(y_train_group)
y_val_data = original_to_group[window_builder._build_label_windows(y_val_multi)]
y_test_data = original_to_group[window_builder._build_label_windows(y_test_multi)]

print(f"    Shapes: train={X_train_seq.shape}   |   test={X_test_seq.shape}")

# Use ALL windows (Normal included) for training
X_train_data = {
    "ae_input": X_train_ae,
    "cnn_input": X_train_seq,
    "lstm_input": X_train_seq,
}

X_val_data = {
    "ae_input": X_val_ae,
    "cnn_input": X_val_seq,
    "lstm_input": X_val_seq,
}

print(f"\n Training data: {len(y_train_data):,} windows")
print(f" Validation data: {len(y_val_data):,} windows")

print("\n calculating adaptative alpha")

counts_orig = pd.Series(y_train_data).value_counts().sort_index()
total_orig = counts_orig.sum()
alpha_raw = 1.0 - (counts_orig / total_orig)
alpha_adaptive = (alpha_raw / alpha_raw.sum()).values.tolist()

class_alpha_list = [CLASS_ALPHA.get(i, 0.25) for i in range(NUM_GROUPS)]

print("\n building model...")
full_model = IDSModelFactory.create_model(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    num_classes=NUM_GROUPS,
    head_depth=HEAD_DEPTH,
    **MODEL_CONFIG,
)

full_model.compile(
    optimizer=Adam(learning_rate=HEAD_LR),
    metrics={
        "classification": ["accuracy", Precision(name="precision"), Recall(name="recall"), F1Score(name="f1_score", average="macro"), AUC(name="auc")]
    },
    loss={
        "classification": CategoricalFocalCrossentropy(gamma=FOCAL_GAMMA, alpha=class_alpha_list),
        "reconstruction": "mse",
    },
    loss_weights={
        "classification": 1.0,
        "reconstruction": 0.0,
    }
)

y_train_ohe = to_categorical(y_train_data, num_classes=NUM_GROUPS)
y_val_ohe = to_categorical(y_val_data, num_classes=NUM_GROUPS)

f1_callbacks = F1EarlyStopping(
    validation_data=(
        X_val_data,
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_data["ae_input"]
        },
    ),
    patience=PATIENCE,
)

checkpoint = ModelCheckpoint(
    filepath=str(MODELS_PATH/"best_model_multiclass.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print(f"\n Training with hybrid mix + aggressive rates...")

train_dataset, n_samples = create_hybrid_mix_dataset(
    X_train_data["ae_input"],
    X_train_data["cnn_input"],
    y_train_data,
    oversample_rates=OVERSAMPLE_RATES,
    original_ratio=0.75,
    jitter_std=JITTER_STD,
    batch_size=BATCH_SIZE,
)

steps_per_epoch = n_samples // BATCH_SIZE

del X_train_data, y_train_data
import gc
gc.collect()

print("preparing test data...")

X_test_data = {
    "ae_input": X_test_ae,
    "cnn_input": X_test_seq,
    "lstm_input": X_test_seq,
}
y_test_ohe = to_categorical(y_test_data, num_classes=NUM_GROUPS)

# Single model training (no ensemble)
history = full_model.fit(
    train_dataset,
    validation_data=(
        X_val_data,
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_data["ae_input"],
        },
    ),
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    shuffle=False,
    callbacks=[checkpoint, reduce_lr],
    verbose=1,
)

print("evaluating model...")

y_pred_probs = full_model.predict(X_test_data, verbose=0)["classification"]
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(
    y_true=y_test_data,
    y_pred=y_pred,
    target_names=GROUP_NAMES,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH / "classification_report.csv")

print("CLASSIFICATION REPORT (All classes)")
print(report_df.to_string())

print(f"\n{'='*60}")
print(f"    Single-Stage Results")
print(f"    Accuracy: {report['accuracy']:.4f}")
print(f"    Macro F1: {report['macro avg']['f1-score']:.4f}")
print(f"\n{'='*60}\n")
