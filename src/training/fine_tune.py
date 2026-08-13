import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
from collections import OrderedDict

import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight

from src.inference import load_model_safe, load_preprocessor
from src.preprocessing.windowing.windowing import WindowGenerator
from src.config import CIC_STAGE1, CIC_STAGE2, FINE_TUNED, FINETUNE_DATA_DIR, UNSW_MODEL

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Configuration per dataset ──────────────────────────────────

DATASET_CONFIGS = {
    "CIC": {
        "window_size": 5,
        "n_features": 41,
        "data_dir": FINETUNE_DATA_DIR,
        "label_map_file": f"{FINETUNE_DATA_DIR}/label_map.json",
        "stages": OrderedDict([
            ("stage1", {
                "source": CIC_STAGE1,
                "save": FINE_TUNED["cic_stage1"],
                "num_classes": 2,
                "head_layers": ["dense", "classification"],
                "label_field": "stage1",
                "label_map": {"BENIGN": 0, "Attack": 1},
            }),
            ("stage2", {
                "source": CIC_STAGE2,
                "save": FINE_TUNED["cic_stage2"],
                "num_classes": 2,
                "head_layers": ["dense", "classification"],
                "label_field": "stage2",
                "label_map": {"Flood": 0, "Rare": 1},
            }),
        ]),
    },
    "UNSW": {
        "window_size": 10,
        "n_features": 193,
        "data_dir": FINETUNE_DATA_DIR,
        "label_map_file": f"{FINETUNE_DATA_DIR}/label_map.json",
        "stages": OrderedDict([
            ("single_stage", {
                "source": UNSW_MODEL,
                "save": FINE_TUNED["unsw"],
                "num_classes": 6,
                "head_layers": ["dense", "classification"],
                "label_field": "group",
                "label_map": {"Normal": 0, "Generic": 1, "Exploits": 2, "Fuzzers": 3,
                              "Recon-Shellcode": 4, "Medium": 5, "Worms": 6},
            }),
        ]),
    },
}


def load_feature_csv(path: str, n_features: int) -> np.ndarray:
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    if len(feat_cols) != n_features:
        print(f"  WARNING: {path}: expected {n_features} features, found {len(feat_cols)}")
    return df[feat_cols].values.astype(np.float32)


def build_finetune_data(
    data_dir: str,
    label_map: dict,
    cfg: dict,
    stage_cfg: dict,
) -> tuple:
    window_size = cfg["window_size"]
    n_features = cfg["n_features"]
    label_field = stage_cfg["label_field"]
    label_map_dict = stage_cfg["label_map"]

    wg = WindowGenerator(window_size=window_size, step=1, pure_windows_only=False)

    all_X_ae = []
    all_X_seq = []
    all_y = []

    for filename, labels in label_map.items():
        if label_field not in labels:
            continue
        label_str = labels[label_field]
        label_idx = label_map_dict.get(label_str)
        if label_idx is None:
            print(f"  SKIP {filename}: unknown label '{label_str}'")
            continue

        filepath = Path(data_dir) / filename
        if not filepath.is_file():
            print(f"  SKIP {filename}: file not found")
            continue

        print(f"  Loading {filename} → label '{label_str}' (idx={label_idx})")
        X = load_feature_csv(str(filepath), n_features)
        if len(X) < window_size:
            print(f"    Too few samples ({len(X)} < window_size {window_size}), skipping")
            continue

        y = np.full(len(X), label_idx, dtype=np.int32)
        X_ae, X_seq, _ = wg.transform(X, y)
        y_w = wg._build_label_windows(y)

        all_X_ae.append(X_ae)
        all_X_seq.append(X_seq)
        all_y.append(y_w)

    if not all_X_ae:
        raise ValueError(f"No data found for label_field='{label_field}'")

    X_ae = np.vstack(all_X_ae)
    X_seq = np.vstack(all_X_seq)
    y = np.concatenate(all_y)

    # Shuffle
    idx = np.random.RandomState(42).permutation(len(y))
    X_ae = X_ae[idx]
    X_seq = X_seq[idx]
    y = y[idx]

    # Split 80/20
    split = int(len(y) * 0.8)
    train = (X_ae[:split], X_seq[:split], y[:split])
    val = (X_ae[split:], X_seq[split:], y[split:])

    print(f"  Train: {len(train[2])} windows, Val: {len(val[2])} windows")
    for name, label in label_map_dict.items():
        cnt = int((y == label).sum())
        print(f"    {name}: {cnt} ({cnt/len(y)*100:.0f}%)")

    return train, val


def freeze_classification_head(model: keras.Model, head_layers: list):
    for layer in model.layers:
        layer.trainable = False
    for name in head_layers:
        try:
            model.get_layer(name).trainable = True
        except ValueError:
            print(f"  WARNING: layer '{name}' not found in model, skipping")


def fine_tune_stage(stage_name: str, stage_cfg: dict, cfg: dict, data_dir: str):
    print(f"\n{'='*60}")
    print(f"  Fine-tuning {stage_name}")
    print(f"{'='*60}")
    print(f"  Source: {stage_cfg['source']}")
    print(f"  Save:   {stage_cfg['save']}")

    # Load label map
    label_map_path = Path(cfg["label_map_file"])
    if not label_map_path.is_file():
        print(f"  ERROR: label_map not found: {label_map_path}")
        return
    with open(label_map_path) as f:
        label_map = json.load(f)

    # Build dataset
    print("\n  Loading feature CSVs...")
    train, val = build_finetune_data(
        data_dir, label_map, cfg, stage_cfg
    )

    X_train_ae, X_train_seq, y_train = train
    X_val_ae, X_val_seq, y_val = val

    # Load pre-trained model
    print(f"\n  Loading pre-trained model...")
    model = load_model_safe(stage_cfg["source"])

    # Freeze feature extractors, keep only classification head
    freeze_classification_head(model, stage_cfg["head_layers"])
    trainable = sum(1 for l in model.layers if l.trainable)
    total = len(model.layers)
    print(f"  Frozen: {total - trainable}/{total} layers, Trainable: {trainable}/{total}")

    # One-hot encode labels
    num_classes = stage_cfg["num_classes"]
    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_val_ohe = to_categorical(y_val, num_classes=num_classes)

    # Class weights to prevent majority class bias (use sample_weight for multi-output)
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"  Class weights: {class_weight_dict}")
    sample_weight_arr = np.ones(len(y_train), dtype=np.float32)
    for cls, w in class_weight_dict.items():
        sample_weight_arr[y_train == cls] = w
    sample_weight = {"classification": sample_weight_arr, "reconstruction": sample_weight_arr}

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss={
            "classification": "categorical_crossentropy",
            "reconstruction": "mse",
        },
        loss_weights={
            "classification": 1.0,
            "reconstruction": 0.0,
        },
        metrics={
            "classification": ["accuracy", Precision(name="precision"), Recall(name="recall")],
        },
    )

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ]

    # Train
    print(f"\n  Training classification head...")
    history = model.fit(
        x={"ae_input": X_train_ae, "cnn_input": X_train_seq, "lstm_input": X_train_seq},
        y={"classification": y_train_ohe, "reconstruction": X_train_ae},
        validation_data=(
            {"ae_input": X_val_ae, "cnn_input": X_val_seq, "lstm_input": X_val_seq},
            {"classification": y_val_ohe, "reconstruction": X_val_ae},
        ),
        epochs=10,
        batch_size=16,
        sample_weight=sample_weight,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_loss = min(history.history["val_loss"])
    best_val_acc = max(history.history.get("val_classification_accuracy", [0]))
    print(f"\n  Best val_loss: {best_val_loss:.4f}, Best val_acc: {best_val_acc:.4f}")

    # Save
    save_path = Path(stage_cfg["save"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"  Model saved: {save_path}")

    # Verify prediction on training data
    print(f"\n  Quick check — predictions on training data:")
    preds = model.predict(
        {"ae_input": X_train_ae[:10], "cnn_input": X_train_seq[:10], "lstm_input": X_train_seq[:10]},
        verbose=0,
    )
    pred_classes = np.argmax(preds["classification"], axis=1)
    print(f"    True:  {y_train[:10]}")
    print(f"    Pred:  {pred_classes}")

    del model, X_train_ae, X_train_seq, y_train, X_val_ae, X_val_seq, y_val
    gc.collect()

    return history


def main():
    parser = argparse.ArgumentParser(description="Fine-tune classification head on labeled real traffic")
    parser.add_argument("--dataset", choices=["CIC", "UNSW"], required=True)
    parser.add_argument("--data-dir", default=FINETUNE_DATA_DIR,
                        help="Directory containing feature CSVs and label_map.json")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    print(f"Fine-tuning {args.dataset}")
    print(f"Data directory: {data_dir}")

    for stage_name, stage_cfg in cfg["stages"].items():
        fine_tune_stage(stage_name, stage_cfg, cfg, str(data_dir))

    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete for {args.dataset}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
