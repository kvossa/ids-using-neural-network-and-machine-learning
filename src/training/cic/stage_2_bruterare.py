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
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.model import IDSModelFactory
from src.preprocessing.windowing.windowing import WindowGenerator
from src.utils.batch_balancer import create_hybrid_mix_dataset, create_class_balanced_dataset
from src.utils.stage1_binary_scoring import apply_stage1_attack_score
from src.utils.train_stopper import F1EarlyStopping
from src.grouping.definitions import (
    CIC_BRUTERARE_MAP, CIC_BRUTERARE_NAMES,
    CIC_BRUTERARE_OVERSAMPLE_RATES, CIC_BRUTERARE_CLASS_ALPHA,
    build_group_mapping,
)

# Set all random seeds
import random
random.seed(42)
np.random.seed(42)
import gc

# Hierarchical grouping config — 7 attack types → 2 groups
# Flood (DDoS + DoS), Rare (Bruteforce+Botnet+Infiltration+Portscan+WebAttacks)
OVERSAMPLE_RATES = CIC_BRUTERARE_OVERSAMPLE_RATES
CLASS_ALPHA = CIC_BRUTERARE_CLASS_ALPHA

# JITTER settings
JITTER_STD = 0.005
ORIGINAL_MIX_RATIO = 0.7

# Training config
HEAD_DEPTH = "attention"
WINDOW_SIZE = 5
WINDOW_STEP = 1
EPOCHS = 50
BATCH_SIZE = 128
PATIENCE = 10
LEARNING_RATE = 5e-5
HEAD_LR = 1e-3
FOCAL_GAMMA = 3.0

# Model architecture config
MODEL_CONFIG = {
    "conv_l2": 0.001,
    "use_first_bn": False,
    "bn_momentum": 0.9,
    "rnn_type": "lstm",
    "rnn_units": 64,
    "rnn_layers": 2,
}
MODEL_TAG = "bruterare"

DROP_COLUMNS = ["Label", "attack_label", "attack_type", "source_file"]
LABEL_COLUMN = "attack_type"
NORMAL_LABEL = "BENIGN"

THRESHOLD = 0.3
STAGE1_DIR = "models/classification/two_stage/cic/stage1"
STAGE1_MODEL = Path(STAGE1_DIR) / "best_model_binary.keras"
STAGE1_THRESHOLD = Path(STAGE1_DIR) / "threshold.json"

REPORTS_PATH = Path("reports/metrics/cic/two_stage/stage") / MODEL_TAG
FIGURES_PATH = Path("reports/figures/cic/two_stage/stage2") / MODEL_TAG
MODELS_PATH = Path("models/classification/two_stage/cic/stage2") / MODEL_TAG

for p in [REPORTS_PATH, FIGURES_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

#LOAD STAGE 1

print(f"\n{'='*60}")
print(f"    IDS Stage 2 - Bruteforce->Rare Regroup (Flood vs Rare)")
print(f"\n{'='*60}\n")

print("loading stage 1 model...")

stage1_model = load_model(STAGE1_MODEL, compile=False)

with open(STAGE1_THRESHOLD) as f:
    threshold_data = json.load(f)

THRESHOLD = threshold_data["threshold"]
cal_tag = threshold_data.get("calibration", "none")
print(f"    Stage 1 threshold: {THRESHOLD} (calibration={cal_tag})")

#LOADING

print("loading data...")

train_df = pd.read_parquet("data/processed/CIC-IDS2017/splits/train/data.parquet")
test_df = pd.read_parquet("data/processed/CIC-IDS2017/splits/test/data.parquet")
val_df = pd.read_parquet("data/processed/CIC-IDS2017/splits/val/data.parquet")

#MULTICLASS LABELS — extract before freeing DataFrames

label_encoder = joblib.load("models/preprocessing/multiclass/cic/label_encoder.pkl")

y_train_multi = label_encoder.transform(train_df[[LABEL_COLUMN]]).ravel()
y_test_multi = label_encoder.transform(test_df[[LABEL_COLUMN]]).ravel()
y_val_multi = label_encoder.transform(val_df[[LABEL_COLUMN]]).ravel()

NUM_CLASSES = len(label_encoder.classes_)
attack_classes = [c for c in label_encoder.classes_ if c != NORMAL_LABEL]

print(f"NUM_CLASSES: {NUM_CLASSES}")
print(f"Attack classes: {attack_classes}")

# Build group mapping from original class indices → group indices
GROUP_NAMES = CIC_BRUTERARE_NAMES
original_to_group = build_group_mapping(
    label_encoder, CIC_BRUTERARE_MAP, GROUP_NAMES, NORMAL_LABEL
)
NUM_GROUPS = len(GROUP_NAMES)
print(f"Groups ({NUM_GROUPS}): {GROUP_NAMES}")
for cls_name in attack_classes:
    orig_idx = list(label_encoder.classes_).index(cls_name)
    grp_idx = original_to_group[orig_idx]
    print(f"  {cls_name} (idx={orig_idx}) → {GROUP_NAMES[grp_idx]} (idx={grp_idx})")

# Binary labels as standalone numpy
y_train_bin = (train_df[LABEL_COLUMN] != NORMAL_LABEL).astype(int).values
y_test_bin = (test_df[LABEL_COLUMN] != NORMAL_LABEL).astype(int).values
y_val_bin = (val_df[LABEL_COLUMN] != NORMAL_LABEL).astype(int).values

# Drop columns from DataFrames
X_train = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])
X_val = val_df.drop(columns=[c for c in DROP_COLUMNS if c in val_df.columns])

# Free original DataFrames
del train_df, test_df, val_df
gc.collect()

#PREPROCESSING 

print("Preprocessing...")
preprocessor = joblib.load("models/preprocessing/binary/cic/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_val_proc = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]

# Free raw DataFrames
del X_train, X_test, X_val
gc.collect()

# ── Pre-windowing ADASYN disabled for CIC (1.6M rows → OOM) ──

#WINDOWING
print("windowing...")

window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=False)

X_train_ae, X_train_seq, y_train_w = window_builder.transform(X_train_proc, y_train_bin)
X_test_ae, X_test_seq, y_test_w = window_builder.transform(X_test_proc, y_test_bin)
X_val_ae, X_val_seq, y_val_w = window_builder.transform(X_val_proc, y_val_bin)

y_train_multi_w = window_builder._build_label_windows(y_train_multi)
y_val_multi_w = window_builder._build_label_windows(y_val_multi)
y_test_multi_w = window_builder._build_label_windows(y_test_multi)

# Free intermediate arrays
del X_train_proc, X_test_proc, X_val_proc, y_train_bin, y_test_bin, y_val_bin
del y_train_multi, y_test_multi, y_val_multi
gc.collect()

print(f"    Shapes: train={X_train_seq.shape}   |   test={X_test_seq.shape}")

#FILTER ATTACKS — chunked to avoid OOM

CHUNK_SIZE = 30000

def chunked_predict_and_score(model, ae, seq, dir, threshold_data, chunk_size=CHUNK_SIZE):
    n = len(ae)
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = {"ae_input": ae[start:end], "cnn_input": seq[start:end], "lstm_input": seq[start:end]}
        probs = model.predict(chunk, verbose=0)["classification"][:, 1]
        scores[start:end] = apply_stage1_attack_score(probs, dir, threshold_data)
        gc.collect()
    return scores

y_train_scores = chunked_predict_and_score(stage1_model, X_train_ae, X_train_seq, STAGE1_DIR, threshold_data)
y_val_scores = chunked_predict_and_score(stage1_model, X_val_ae, X_val_seq, STAGE1_DIR, threshold_data)

attack_train_mask = y_train_scores > THRESHOLD
attack_val_mask = y_val_scores > THRESHOLD

X_train_attack = {
    k: v[attack_train_mask] for k, v in {
        "ae_input": X_train_ae,
        "cnn_input": X_train_seq,
        "lstm_input": X_train_seq,
    }.items()
}

y_train_attack = y_train_multi_w[attack_train_mask]
# Remap multiclass labels to group indices and filter out BENIGN (-1)
y_train_attack = original_to_group[y_train_attack]
train_keep = y_train_attack >= 0
X_train_attack = {k: v[train_keep] for k, v in X_train_attack.items()}
y_train_attack = y_train_attack[train_keep]

X_val_attack = {
    k: v[attack_val_mask] for k, v in {
        "ae_input": X_val_ae,
        "cnn_input": X_val_seq,
        "lstm_input": X_val_seq,
    }.items()
}

y_val_attack = y_val_multi_w[attack_val_mask]
# Remap and filter val
y_val_attack = original_to_group[y_val_attack]
val_keep = y_val_attack >= 0
X_val_attack = {k: v[val_keep] for k, v in X_val_attack.items()}
y_val_attack = y_val_attack[val_keep]

print(f"    TRAIN ATTACKS: {sum(attack_train_mask):,} ({sum(attack_train_mask)/len(attack_train_mask)*100:.1f}%)")
print(f"    VAL ATTACKS: {sum(attack_val_mask):,} ({sum(attack_val_mask)/len(attack_val_mask)*100:.1f}%)")

# Free full-window arrays — only attack subsets needed from now on
del X_train_ae, X_train_seq, X_val_ae, X_val_seq
del y_train_w, y_val_w, y_train_multi_w, y_val_multi_w
del y_train_scores, y_val_scores, attack_train_mask, attack_val_mask
gc.collect()

#CREATE STAGE 2

print(f"\n building stage 2 model with frozen encoder...")

print("\n calculating adaptative alpha")

counts_orig = pd.Series(y_train_attack).value_counts().sort_index()
total_orig = counts_orig.sum()

alpha_raw = 1.0 - (counts_orig / total_orig)
alpha_adaptive = (alpha_raw / alpha_raw.sum()).values.tolist()

class_alpha_list = [CLASS_ALPHA.get(i, 0.25) for i in range(NUM_GROUPS)]

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

# full_model.summary()

y_train_ohe = to_categorical(y_train_attack, num_classes=NUM_GROUPS)
y_val_ohe = to_categorical(y_val_attack, num_classes=NUM_GROUPS)


f1_callbacks = F1EarlyStopping(
    validation_data=(
        X_val_attack,
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_attack["ae_input"]
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


#TRAINING

print(f"\n Training stage 2 with hybrid mix + aggressive rates...")

train_dataset, n_samples = create_hybrid_mix_dataset(
    X_train_attack["ae_input"],
    X_train_attack["cnn_input"],
    y_train_attack,
    oversample_rates=OVERSAMPLE_RATES,
    original_ratio=ORIGINAL_MIX_RATIO,
    jitter_std=JITTER_STD,
    batch_size=BATCH_SIZE,
)

steps_per_epoch = n_samples // BATCH_SIZE

# Free training arrays — data is now inside tf.data dataset
del X_train_attack, y_train_attack
gc.collect()

history = full_model.fit(
    train_dataset,
    validation_data=(
        X_val_attack,
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_attack["ae_input"],
        },
    ),
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    shuffle=False,
    callbacks=[checkpoint, reduce_lr],
    verbose=1,
)

# Free validation arrays (no longer needed after training)
del X_val_attack, y_val_attack, y_val_ohe
gc.collect()

# EVALUATION

print("evaluating model...")

y_test_scores = chunked_predict_and_score(stage1_model, X_test_ae, X_test_seq, STAGE1_DIR, threshold_data)
attack_test_mask = y_test_scores > THRESHOLD


X_test_attack = {
    k: v[attack_test_mask] for k, v in {
        "ae_input": X_test_ae,
        "cnn_input": X_test_seq,
        "lstm_input": X_test_seq,
    }.items()
}

y_test_attack = y_test_multi_w[attack_test_mask]
# Remap test labels to groups and filter out BENIGN
y_test_attack = original_to_group[y_test_attack]
test_keep = y_test_attack >= 0
X_test_attack = {k: v[test_keep] for k, v in X_test_attack.items()}
y_test_attack = y_test_attack[test_keep]
y_test_ohe = to_categorical(y_test_attack, num_classes=NUM_GROUPS)

# Free full window arrays (no longer needed)
del X_test_ae, X_test_seq, y_test_multi_w
gc.collect()

y_pred_probs = full_model.predict(X_test_attack, verbose=0)["classification"]
y_pred = np.argmax(y_pred_probs, axis=1)

# === EVALUATION ===
report = classification_report(
    y_true=y_test_attack,
    y_pred=y_pred,
    target_names=GROUP_NAMES,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH / "classification_report.csv")

print("CLASSIFICATION REPORT (Attack types only)")
print(report_df.to_string())

print(f"\n{'='*60}")
print(f"    Stage 2 Results (Attack Classification)")
print(f"    Accuracy: {report['accuracy']:.4f}")
print(f"    Macro F1: {report['macro avg']['f1-score']:.4f}")
print(f"\n{'='*60}\n")
