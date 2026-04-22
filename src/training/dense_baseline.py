"""
================================================================================
Phase 1c: Fast Dense Network (No LSTM) - Quick Baseline
================================================================================
Goal: Fast baseline to establish that NN can learn
Architecture: Flat input -> Dense Network (no temporal)
Success Criteria: Macro F1 > 0.50
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
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, F1Score, AUC
from sklearn.metrics import classification_report
from collections import Counter

# ==============================================================================
# FAST CONFIG
# ==============================================================================

EPOCHS = 20
BATCH_SIZE = 512
PATIENCE = 5
LEARNING_RATE = 1e-3

REPORTS_PATH = Path("reports/metrics/unsw/multiclass/dense_baseline")
MODELS_PATH = Path("models/classification/multiclass/unsw/dense_baseline")

for p in [REPORTS_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. LOAD AND PREPROCESS
# ==============================================================================

print("\n" + "=" * 60)
print("PHASE 1c: DENSE NETWORK BASELINE (FAST)")
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

print("\n[2] Preprocessing...")
preprocessor = joblib.load("models/preprocessing/multiclass/unsw/preprocessing.pkl")
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_val_proc = preprocessor.transform(X_val)

label_encoder = joblib.load("models/preprocessing/multiclass/unsw/label_encoder.pkl")
y_train_enc = label_encoder.transform(y_train_raw)
y_test_enc = label_encoder.transform(y_test_raw)
y_val_enc = label_encoder.transform(y_val_raw)

NUM_CLASSES = len(label_encoder.classes_)
class_names = label_encoder.classes_.tolist()
num_features = X_train_proc.shape[1]

print(f"    Features: {num_features}, Train: {X_train_proc.shape[0]}, Test: {X_test_proc.shape[0]}")

# ==============================================================================
# 2. CLASS WEIGHTS
# ==============================================================================

print("\n[3] Class weights...")
class_counts = Counter(y_train_enc)
total = sum(class_counts.values())
class_weights = {i: total / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)}

# ==============================================================================
# 3. BUILD DENSE MODEL
# ==============================================================================

print("\n[4] Building Dense model...")

inputs = Input(shape=(num_features,), name='input')
x = Dense(256, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy', Precision(), Recall(), F1Score(average='macro'), AUC()],
    loss='categorical_crossentropy',
)

model.summary()

# ==============================================================================
# 4. TRAIN
# ==============================================================================

print("\n[5] Training...")

y_train_ohe = keras.utils.to_categorical(y_train_enc, NUM_CLASSES)
y_val_ohe = keras.utils.to_categorical(y_val_enc, NUM_CLASSES)
y_test_ohe = keras.utils.to_categorical(y_test_enc, NUM_CLASSES)

early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

history = model.fit(
    X_train_proc.values, y_train_ohe,
    validation_data=(X_val_proc.values, y_val_ohe),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=2,
)

# ==============================================================================
# 5. EVALUATE
# ==============================================================================

print("\n[6] Evaluating...")

y_pred = np.argmax(model.predict(X_test_proc.values, verbose=0), axis=1)

report = classification_report(y_test_enc, y_pred, target_names=class_names, digits=3, zero_division=0, output_dict=True)
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (Dense Baseline)")
print("=" * 60)
print(classification_report(y_test_enc, y_pred, target_names=class_names, digits=3, zero_division=0))

macro_f1 = report['macro avg']['f1-score']
print(f"\n>>> Macro F1: {macro_f1:.3f}")
print("✓ SUCCESS" if macro_f1 > 0.50 else "✗ FAILED")

# Save
pd.DataFrame(report).transpose().to_csv(REPORTS_PATH / "report.csv")
pd.DataFrame([{'macro_f1': macro_f1, 'epochs': len(history.history['loss'])}]).to_csv(REPORTS_PATH / "summary.csv", index=False)
print(f"\nSaved to {REPORTS_PATH}")