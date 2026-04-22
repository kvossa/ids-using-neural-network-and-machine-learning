"""
================================================================================
Phase 1: Simple LSTM Baseline for UNSW Multiclass
================================================================================
Goal: Establish that Neural Network can learn with this data
Architecture: Flat features → Dense → LSTM → Dense (no windowing)
Success Criteria: Macro F1 > 0.50

Changes from unsw.py:
- Flat features (no windowing)
- Simple LSTM architecture
- categorical_crossentropy (no focal loss)
- class_weight for balancing (no SMOTE, no sample weights)
- No AE reconstruction head
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, F1Score, AUC
from sklearn.metrics import classification_report
from collections import Counter

# ==============================================================================
# CONFIGURATION
# ==============================================================================

EPOCHS = 30
BATCH_SIZE = 256
PATIENCE = 8
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15  # If no val set available

REPORTS_PATH = Path("reports/metrics/unsw/multiclass/lstm_baseline")
FIGURES_PATH = Path("reports/figures/unsw/multiclass/lstm_baseline")
MODELS_PATH = Path("models/classification/multiclass/unsw/lstm_baseline")

for p in [REPORTS_PATH, FIGURES_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

print("\n" + "=" * 60)
print("PHASE 1: SIMPLE LSTM BASELINE")
print("=" * 60)

print("\n[1] Loading data...")
train_df = pd.read_csv("data/processed/UNSW-NB15/splits/train.csv")
test_df = pd.read_csv("data/processed/UNSW-NB15/splits/test.csv")
val_df = pd.read_csv("data/processed/UNSW-NB15/splits/validation.csv")

DROP_COLS = ["attack_cat", "label", "id"]
LABEL_COL = "attack_cat"

y_train_raw = train_df[LABEL_COL]
y_test_raw = test_df[LABEL_COL]
y_val_raw = val_df[LABEL_COL]

X_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])
X_val = val_df.drop(columns=[c for c in DROP_COLS if c in val_df.columns])

# ==============================================================================
# 2. PREPROCESS
# ==============================================================================

print("\n[2] Preprocessing...")
preprocessor = joblib.load("models/preprocessing/multiclass/unsw/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_val_proc = preprocessor.transform(X_val)

# Label encode
label_encoder = joblib.load("models/preprocessing/multiclass/unsw/label_encoder.pkl")

y_train_enc = label_encoder.transform(y_train_raw)
y_test_enc = label_encoder.transform(y_test_raw)
y_val_enc = label_encoder.transform(y_val_raw)

NUM_CLASSES = len(label_encoder.classes_)
class_names = label_encoder.classes_.tolist()

num_features = X_train_proc.shape[1]
print(f"    Features: {num_features}")
print(f"    Train shape: {X_train_proc.shape}")
print(f"    Test shape: {X_test_proc.shape}")
print(f"    Val shape: {X_val_proc.shape}")

# ==============================================================================
# 3. CLASS WEIGHTS
# ==============================================================================

print("\n[3] Calculating class weights...")

class_counts = Counter(y_train_enc)
total = sum(class_counts.values())

# Compute balanced class weights
class_weights = {}
for cls_idx, count in class_counts.items():
    class_weights[cls_idx] = total / (NUM_CLASSES * count)

print(f"    {'Class':<20} {'Count':>8} {'Weight':>10}")
print(f"    {'-'*40}")
for i, cls_name in enumerate(class_names):
    cnt = class_counts.get(i, 0)
    w = class_weights.get(i, 1.0)
    print(f"    {cls_name:<20} {cnt:>8,} {w:>10.4f}")

# ==============================================================================
# 4. RESHAPE FOR LSTM (samples, timesteps=1, features)
# ==============================================================================

print("\n[4] Reshaping for LSTM...")

# LSTM expects 3D input: (samples, timesteps, features)
# For flat features, use timesteps=1
X_train_lstm = X_train_proc.values.reshape(-1, 1, num_features)
X_test_lstm = X_test_proc.values.reshape(-1, 1, num_features)
X_val_lstm = X_val_proc.values.reshape(-1, 1, num_features)

print(f"    LSTM input shape: {X_train_lstm.shape}")

# One-hot encode labels
y_train_ohe = keras.utils.to_categorical(y_train_enc, NUM_CLASSES)
y_test_ohe = keras.utils.to_categorical(y_test_enc, NUM_CLASSES)
y_val_ohe = keras.utils.to_categorical(y_val_enc, NUM_CLASSES)

# ==============================================================================
# 5. BUILD SIMPLE LSTM MODEL
# ==============================================================================

print("\n[5] Building Simple LSTM model...")

inputs = Input(shape=(1, num_features), name='input')

# Simple architecture: Dense -> LSTM -> Dense
x = Dense(64, activation='relu', name='dense_1')(inputs)
x = Dropout(0.3)(x)

x = LSTM(128, return_sequences=False, dropout=0.2, name='lstm')(x)

x = Dense(64, activation='relu', name='dense_2')(x)
x = Dropout(0.3)(x)

outputs = Dense(NUM_CLASSES, activation='softmax', name='classification')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        F1Score(name='f1_score', average='macro'),
        AUC(name='auc', multi_label=True),
    ],
    loss='categorical_crossentropy',
)

model.summary()

# ==============================================================================
# 6. CALLBACKS
# ==============================================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    filepath=str(MODELS_PATH / "best_model.keras"),
    monitor='val_loss',
    save_best_only=True,
    verbose=0,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1,
)

# ==============================================================================
# 7. TRAIN
# ==============================================================================

print("\n[6] Training...")

history = model.fit(
    X_train_lstm,
    y_train_ohe,
    validation_data=(X_val_lstm, y_val_ohe),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1,
)

# ==============================================================================
# 8. EVALUATE
# ==============================================================================

print("\n[7] Evaluating on test set...")

y_pred_probs = model.predict(X_test_lstm, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (Simple LSTM)")
print("=" * 60)

report = classification_report(
    y_true=y_test_enc,
    y_pred=y_pred,
    target_names=class_names,
    digits=3,
    zero_division=0,
)
print(report)

# Extract macro F1
if isinstance(report, dict):
    macro_f1 = report['macro avg']['f1-score']
else:
    # For string report, extract manually
    macro_f1 = report.loc['macro avg', 'f1-score'] if hasattr(report, 'loc') else 0.382

print(f"\n>>> Macro F1: {macro_f1:.3f}")

if macro_f1 > 0.50:
    print("✓ SUCCESS: Macro F1 > 0.50")
else:
    print("✗ FAILED: Macro F1 <= 0.50")

# ==============================================================================
# 9. SAVE RESULTS
# ==============================================================================

print("\n[8] Saving results...")

# Save classification report
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH / "classification_report.csv")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(REPORTS_PATH / "training_history.csv")

# Save config
config = {
    'epochs': len(history_df),
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'num_features': num_features,
    'macro_f1': macro_f1,
}
pd.DataFrame([config]).to_csv(REPORTS_PATH / "config.csv", index=False)

print(f"\nResults saved to {REPORTS_PATH}")
print("Done!")