import joblib
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
from sklearn.metrics import classification_report

from src.model.model import IDSModelFactory
from src.preprocessing.windowing.windowing import WindowGenerator
from src.utils.train_stopper import F1EarlyStopping
from src.grouping.definitions import (
    CIC_SINGLE_GROUP_MAP, CIC_SINGLE_GROUP_NAMES,
    CIC_SINGLE_OVERSAMPLE_RATES, CIC_SINGLE_CLASS_ALPHA,
    build_group_mapping,
)

import random
random.seed(42)
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────
GROUP_MAP = CIC_SINGLE_GROUP_MAP
GROUP_NAMES = list(CIC_SINGLE_GROUP_NAMES)
OVERSAMPLE_RATES = CIC_SINGLE_OVERSAMPLE_RATES
CLASS_ALPHA = CIC_SINGLE_CLASS_ALPHA

HEAD_DEPTH = "attention"
WINDOW_SIZE = 5
WINDOW_STEP = 1
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 10
HEAD_LR = 1e-3
FOCAL_GAMMA = 3.0
JITTER_STD = 0.005
ORIGINAL_MIX_RATIO = 0.75

# Model architecture config (toggle between baseline and experiment)
MODEL_CONFIG = {
    "conv_l2": 0.001,
    "use_first_bn": False,
    "bn_momentum": 0.9,
    "rnn_type": "lstm",
    "rnn_units": 64,
    "rnn_layers": 2,
}
MODEL_TAG = "experiment"

DROP_COLUMNS = ["Label", "attack_label", "attack_type", "source_file"]
LABEL_COLUMN = "attack_type"
NORMAL_LABEL = "BENIGN"

REPORTS_PATH = Path("reports/metrics/cic/single_stage") / MODEL_TAG
FIGURES_PATH = Path("reports/figures/cic/single_stage") / MODEL_TAG
MODELS_PATH = Path("models/classification/single_stage/cic") / MODEL_TAG

for p in [REPORTS_PATH, FIGURES_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"    CIC Single-Stage Multiclass Classification")
print(f"\n{'='*60}\n")

print("loading data...")
train_df = pd.read_parquet("data/processed/CIC-IDS2017/splits/train/data.parquet")
test_df = pd.read_parquet("data/processed/CIC-IDS2017/splits/test/data.parquet")
val_df = pd.read_parquet("data/processed/CIC-IDS2017/splits/val/data.parquet")

y_train_raw = train_df[LABEL_COLUMN]
y_test_raw = test_df[LABEL_COLUMN]
y_val_raw = val_df[LABEL_COLUMN]

X_train = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])
X_val = val_df.drop(columns=[c for c in DROP_COLUMNS if c in val_df.columns])

label_encoder = joblib.load("models/preprocessing/multiclass/cic/label_encoder.pkl")

y_train_multi = label_encoder.transform(y_train_raw)
y_test_multi = label_encoder.transform(y_test_raw)
y_val_multi = label_encoder.transform(y_val_raw)

NUM_CLASSES = len(label_encoder.classes_)
print(f"Original classes: {NUM_CLASSES}")
print(f"Classes: {label_encoder.classes_.tolist()}")

original_to_group = build_group_mapping(
    label_encoder, GROUP_MAP, GROUP_NAMES, normal_label=None
)
NUM_GROUPS = len(GROUP_NAMES)
print(f"Groups ({NUM_GROUPS}): {GROUP_NAMES}")

print("Preprocessing...")
preprocessor = joblib.load("models/preprocessing/binary/cic/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_val_proc = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]

del X_train, X_test, X_val, train_df, test_df, val_df
import gc
gc.collect()

y_train_group = original_to_group[y_train_multi]
y_train_binary = (y_train_group != 0).astype(int)

# ── Windowing (no pre-windowing SMOTE — 1.6M rows OOM) ──────
print("\n windowing...")
window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=False)

X_train_ae, X_train_seq, _ = window_builder.transform(X_train_proc, y_train_binary)
X_test_ae, X_test_seq, _ = window_builder.transform(X_test_proc, (y_test_raw != NORMAL_LABEL).astype(int))
X_val_ae, X_val_seq, _ = window_builder.transform(X_val_proc, (y_val_raw != NORMAL_LABEL).astype(int))

y_train_data = window_builder._build_label_windows(y_train_group)
y_val_data = original_to_group[window_builder._build_label_windows(y_val_multi)]
y_test_data = original_to_group[window_builder._build_label_windows(y_test_multi)]

print(f"    Shapes: train={X_train_seq.shape}   |   test={X_test_seq.shape}")

del X_train_proc, X_test_proc, X_val_proc, y_train_binary, y_train_group
gc.collect()

# Use ALL windows (BENIGN included) for training
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

print("\n calculating class alpha")

counts_orig = pd.Series(y_train_data).value_counts().sort_index()
total_orig = counts_orig.sum()
print("    Distribution:")
for i in range(NUM_GROUPS):
    cnt = counts_orig.get(i, 0)
    print(f"      {GROUP_NAMES[i]:<20s} {cnt:>8,} ({cnt/total_orig*100:.1f}%)")

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

print(f"\n creating tf.data pipeline...")

train_dataset = tf.data.Dataset.from_tensor_slices((
    X_train_data,
    {"classification": y_train_ohe, "reconstruction": X_train_data["ae_input"]}
))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
steps_per_epoch = len(y_train_data) // BATCH_SIZE

# Free source arrays — tf.data keeps internal references
del X_train_ae, X_train_seq, X_train_data, y_train_data, y_train_ohe
import gc
gc.collect()

print("preparing test data...")

X_test_data = {
    "ae_input": X_test_ae,
    "cnn_input": X_test_seq,
    "lstm_input": X_test_seq,
}
y_test_ohe = to_categorical(y_test_data, num_classes=NUM_GROUPS)

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
