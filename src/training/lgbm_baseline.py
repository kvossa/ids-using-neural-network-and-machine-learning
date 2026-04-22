"""
Quick baseline: LGBM for UNSW multiclass
Run this first to verify data pipeline works before using complex neural network
"""
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from collections import Counter

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("Loading data...")
train_df = pd.read_csv("data/processed/UNSW-NB15/splits/train.csv")
test_df  = pd.read_csv("data/processed/UNSW-NB15/splits/test.csv")

DROP_COLS = ["attack_cat", "label", "id"]
LABEL_COL = "attack_cat"

y_train = train_df[LABEL_COL]
y_test  = test_df[LABEL_COL]
X_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
X_test  = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

# ==============================================================================
# 2. PREPROCESS
# ==============================================================================
print("Preprocessing...")
preprocessor = joblib.load("models/preprocessing/multiclass/unsw/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Label encode
label_encoder = joblib.load("models/preprocessing/multiclass/unsw/label_encoder.pkl")
y_train_enc = label_encoder.transform(y_train)
y_test_enc  = label_encoder.transform(y_test)

print(f"Features: {X_train_proc.shape[1]}")
print(f"Train shape: {X_train_proc.shape}")
print(f"Test shape: {X_test_proc.shape}")

# ==============================================================================
# 3. TRAIN LGBM (no windowing, no SMOTE - just raw features)
# ==============================================================================
print("\nTraining LGBM...")

# Calculate class weights
class_counts = Counter(y_train_enc)
total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

model = LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=63,
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model.fit(X_train_proc, y_train_enc)

# ==============================================================================
# 4. EVALUATE
# ==============================================================================
print("\nEvaluating...")
y_pred = model.predict(X_test_proc)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (LGBM Baseline)")
print("=" * 60)

report = classification_report(
    y_true=y_test_enc,
    y_pred=y_pred,
    target_names=label_encoder.classes_,
    digits=3,
    zero_division=0,
)
print(report)

# Save predictions
np.save("reports/metrics/unsw/multiclass/lgbm_predictions.npy", y_pred)
print("\nPredictions saved to reports/metrics/unsw/multiclass/lgbm_predictions.npy")