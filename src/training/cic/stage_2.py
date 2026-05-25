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

# Set all random seeds
import random
random.seed(42)
np.random.seed(42)

# Oversample rates (aggressive for rare classes)
OVERSAMPLE_RATES = {
    0: 30,   # Analysis
    1: 30,   # Backdoor (increased from 8)
    2: 15,   # DoS (increased from 8)
    8: 30,   # Shellcode (increased from 8)
    9: 100,  # Worms (increased from 30)
}

# JITTER settings
JITTER_STD = 0.005
ORIGINAL_MIX_RATIO = 0.7

# Model config
HEAD_DEPTH = "standard"
ENSEMBLE_MODELS = 3  # Test first

# Class-specific focal alpha (higher for rare classes)
CLASS_ALPHA = {
    0: 0.75,  # Analysis (increased)
    1: 0.80,  # Backdoor (increased)
    2: 0.60,  # DoS (increased)
    3: 0.15,  # Exploits
    4: 0.25,  # Fuzzers
    5: 0.15,  # Generic
    6: 0.20,  # Normal
    7: 0.30,  # Reconnaissance
    8: 0.80,  # Shellcode (increased)
    9: 0.90,  # Worms (increased)
}

HIERARCHICAL_GROUPS = {
    0: "ClientAttack",  # Analysis
    1: "ClientAttack",  # Backdoor  
    2: "Flood",         # DoS
    3: "Exploits",       # Exploits
    4: "Flood",         # Fuzzers
    5: "Generic",        # Generic
    6: "Normal",         # Normal
    7: "Reconnaissance", # Reconnaissance
    8: "ClientAttack",  # Shellcode
    9: "Worms",         # Worms
}

GROUP_MAPPING = {
    0: "ClientAttack",
    1: "ClientAttack", 
    2: "Flood",
    3: "Exploits",
    4: "Flood",
    5: "Generic",
    6: "Normal",
    7: "Reconnaissance", 
    8: "ClientAttack",
    9: "Worms",
}

GROUP_TO_IDX = {g: i for i, g in enumerate(set(GROUP_MAPPING.values()))}
IDX_TO_GROUP = {v: k for k, v in GROUP_TO_IDX.items()}

print("\n=== HIERARCHICAL GROUPING ===")
print("Groups:", GROUP_TO_IDX)
for group_name, group_idx in GROUP_TO_IDX.items():
    original_classes = [c for c, g in GROUP_MAPPING.items() if g == group_name]
    print(f"  {group_name} (idx={group_idx}): classes {original_classes}")

WINDOW_SIZE = 1
WINDOW_STEP = 1
EPOCHS = 50
BATCH_SIZE = 128
PATIENCE = 10
LEARNING_RATE = 5e-5
HEAD_LR = 1e-3
FOCAL_GAMMA = 3.0

DROP_COLUMNS = ["Label", "attack_label", "attack_type", "source_file"]
LABEL_COLUMN = "attack_type"
NORMAL_LABEL = "BENIGN"

STAGE1_MODEL = Path("models/classification/two_stage/cic/stage1/best_model_binary.keras")#.h5??
STAGE1_THRESHOLD = Path("models/classification/two_stage/cic/stage1/threshold.json")
STAGE1_DIR = STAGE1_MODEL.parent

REPORTS_PATH = Path("reports/metrics/cic/two_stage/stage")
FIGURES_PATH = Path("reports/figures/cic/two_stage/stage2")
MODELS_PATH = Path("models/classification/two_stage/cic/stage2")

for p in [REPORTS_PATH, FIGURES_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

#LOAD STAGE 1

print(f"\n{'='*60}")
print(f"    IDS Stage 2 -  Multiclass Classification")
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

y_train_raw = train_df[LABEL_COLUMN]
y_test_raw = test_df[LABEL_COLUMN]
y_val_raw = val_df[LABEL_COLUMN]

X_train = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])
X_val = val_df.drop(columns=[c for c in DROP_COLUMNS if c in val_df.columns])

#MULTICLASS LABELS

label_encoder = joblib.load("models/preprocessing/multiclass/cic/label_encoder.pkl")

y_train_multi = label_encoder.transform(train_df[[LABEL_COLUMN]])
y_test_multi = label_encoder.transform(test_df[[LABEL_COLUMN]])
y_val_multi = label_encoder.transform(val_df[[LABEL_COLUMN]])

NUM_CLASSES = len(label_encoder.classes_)
attack_classes = [c for c in label_encoder.classes_ if c != NORMAL_LABEL]

print(f"NUM_CLASSES: {NUM_CLASSES}")

print(f"Attack classes: {attack_classes}")

#PREPROCESSING 

print("Preprocessing...")
preprocessor = joblib.load("models/preprocessing/binary/cic/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_val_proc = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]

#WINDOWING
print("windowing...")

window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=False)

X_train_ae, X_train_seq, y_train_w = window_builder.transform(X_train_proc, (y_train_raw != NORMAL_LABEL).astype(int))
X_test_ae, X_test_seq, y_test_w = window_builder.transform(X_test_proc, (y_test_raw != NORMAL_LABEL).astype(int))
X_val_ae, X_val_seq, y_val_w = window_builder.transform(X_val_proc, (y_val_raw != NORMAL_LABEL).astype(int))

y_train_multi_w = window_builder._build_label_windows(y_train_multi)
y_val_multi_w = window_builder._build_label_windows(y_val_multi)
y_test_multi_w = window_builder._build_label_windows(y_test_multi)

print(f"    Shapes: train={X_train_seq.shape}   |   test={X_test_seq.shape}")

#FILTER ATTACKS

print("filtering attack samples")

y_train_probs = stage1_model.predict(
    {
        "ae_input": X_train_ae,
        "cnn_input": X_train_seq,
        "lstm_input": X_train_seq,
    },
    verbose = 0
)["classification"][:, 1]

y_val_probs = stage1_model.predict(
    {
        "ae_input": X_val_ae,
        "cnn_input": X_val_seq,
        "lstm_input": X_val_seq,
    },
    verbose = 0
)["classification"][:, 1]

y_train_scores = apply_stage1_attack_score(y_train_probs, STAGE1_DIR, threshold_data)
y_val_scores = apply_stage1_attack_score(y_val_probs, STAGE1_DIR, threshold_data)

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

X_val_attack = {
    k: v[attack_val_mask] for k, v in {
        "ae_input": X_val_ae,
        "cnn_input": X_val_seq,
        "lstm_input": X_val_seq,
    }.items()
}

y_val_attack = y_val_multi_w[attack_val_mask]

print(f"    TRAIN ATTACKS: {sum(attack_train_mask):,} ({sum(attack_train_mask)/len(attack_train_mask)*100:.1f}%)")
print(f"    VAL ATTACKS: {sum(attack_val_mask):,} ({sum(attack_val_mask)/len(attack_val_mask)*100:.1f}%)")

#CREATE STAGE 2

print(f"\n building stage 2 model with frozen encoder...")

print("\n calculating adaptative alpha")

counts_orig = pd.Series(y_train_attack).value_counts().sort_index()
total_orig = counts_orig.sum()

alpha_raw = 1.0 - (counts_orig / total_orig)
alpha_adaptive = (alpha_raw / alpha_raw.sum()).values.tolist()

class_alpha_list = [CLASS_ALPHA.get(i, 0.25) for i in range(NUM_CLASSES)]

full_model = IDSModelFactory.create_model(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    num_classes=NUM_CLASSES,
    head_depth=HEAD_DEPTH,
)

stage1_weights = stage1_model.get_weights()

for layer in full_model.layers:
    layer.trainable = False

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

full_model.load_weights(STAGE1_MODEL, skip_mismatch=True)

for layer in full_model.layers:
    layer.trainable = True

print(" Trainable layers:")
for layer in full_model.layers:
    if layer.trainable:
        print(f"    - {layer.name}")

#COMPILE

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

y_train_ohe = to_categorical(y_train_attack, num_classes=NUM_CLASSES)
y_val_ohe = to_categorical(y_val_attack, num_classes=NUM_CLASSES)


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
    original_ratio=0.75,  # More original data to help normalization
    jitter_std=JITTER_STD,
    batch_size=BATCH_SIZE,
)

steps_per_epoch = n_samples // BATCH_SIZE

steps_per_epoch = n_samples // BATCH_SIZE

# === TEST SETUP (needed for ensemble and single) ===
print("preparing test data...")

y_test_probs = stage1_model.predict(
    {
        "ae_input": X_test_ae,
        "cnn_input": X_test_seq,
        "lstm_input": X_test_seq,
    },
    verbose = 0
)["classification"][:, 1]

y_test_scores = apply_stage1_attack_score(y_test_probs, STAGE1_DIR, threshold_data)
attack_test_mask = y_test_scores > THRESHOLD


X_test_attack = {
    k: v[attack_test_mask] for k, v in {
        "ae_input": X_test_ae,
        "cnn_input": X_test_seq,
        "lstm_input": X_test_seq,
    }.items()
}

y_test_attack = y_test_multi_w[attack_test_mask]
y_test_ohe = to_categorical(y_test_attack, num_classes=NUM_CLASSES)

# === ENSEMBLE TRAINING ===
if ENSEMBLE_MODELS > 1:
    print(f"\n=== ENSEMBLE TRAINING ({ENSEMBLE_MODELS} models) ===")
    ensemble_models = []
    ensemble_histories = []
    
    for model_idx in range(ENSEMBLE_MODELS):
        print(f"\n--- Training model {model_idx + 1}/{ENSEMBLE_MODELS} ---")
        
        # Create new model for each ensemble member (fresh, no weight loading)
        full_model = IDSModelFactory.create_model(
            window_size=WINDOW_SIZE,
            num_features=num_features,
            num_classes=NUM_CLASSES,
            head_depth=HEAD_DEPTH,
        )
        
        for layer in full_model.layers:
            layer.trainable = False
        
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
        
        for layer in full_model.layers:
            layer.trainable = True
        
        # Different seed for each model
        model_seed = 42 + model_idx * 100
        tf.random.set_seed(model_seed)
        np.random.seed(model_seed)
        
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
            verbose=0,
        )
        
        ensemble_models.append(full_model)
        ensemble_histories.append(history.history)
        
        # Save each model
        full_model.save(MODELS_PATH / f"ensemble_model_{model_idx}.keras")
        
        # Quick eval
        y_pred = np.argmax(full_model.predict(X_test_attack, verbose=0)["classification"], axis=1)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test_attack, y_pred, average='macro')
        print(f"    Model {model_idx} test F1: {f1:.4f}")
    
    # === ENSEMBLE PREDICTION ===
    print("\n--- Ensemble prediction ---")
    
    y_pred_probs_ensemble = []
    for model_idx, model in enumerate(ensemble_models):
        y_pred_probs_ensemble.append(model.predict(X_test_attack, verbose=0)["classification"])
    
    # Average predictions
    y_pred_probs_avg = np.mean(y_pred_probs_ensemble, axis=0)
    y_pred = np.argmax(y_pred_probs_avg, axis=1)
    
else:
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
        verbose=1,
    )
    
    # EVALUATION
    
    print("evaluating model...")
    
    y_test_probs = stage1_model.predict(
        {
            "ae_input": X_test_ae,
            "cnn_input": X_test_seq,
            "lstm_input": X_test_seq,
        },
        verbose = 0
    )["classification"][:, 1]

    y_test_scores = apply_stage1_attack_score(y_test_probs, STAGE1_DIR, threshold_data)
    attack_test_mask = y_test_scores > THRESHOLD


    X_test_attack = {
        k: v[attack_test_mask] for k, v in {
            "ae_input": X_test_ae,
            "cnn_input": X_test_seq,
            "lstm_input": X_test_seq,
        }.items()
    }

    y_test_attack = y_test_multi_w[attack_test_mask]
    y_test_ohe = to_categorical(y_test_attack, num_classes=NUM_CLASSES)

    y_pred_probs = full_model.predict(X_test_attack, verbose=0)["classification"]
    y_pred = np.argmax(y_pred_probs, axis=1)

# === EVALUATION ===
report = classification_report(
    y_true=y_test_attack,
    y_pred=y_pred,
    target_names=label_encoder.classes_.tolist(),
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
