import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from keras.metrics import AUC, Precision, Recall, F1Score
from keras.utils import to_categorical
from keras.losses import  BinaryFocalCrossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.model import IDSModelFactory
from src.preprocessing.windowing.windowing import WindowGenerator
from src.utils.batch_balancer import create_balanced_tf_dataset
from src.utils.train_stopper import F1EarlyStopping

#CONF

WINDOW_SIZE = 10
WINDOW_STEP = 1
EPOCHS = 50
BATCH_SIZE = 256
PATIENCE = 10
# After balanced base weights; tune if one class still dominates recall.
ATTACK_WEIGHT = 1.0
NORMAL_WEIGHT = 1.0
BINARY_THRESHOLD = 0.3
LEARNING_RATE = 1e-4

DROP_COLUMNS = ["attack_cat", "label", "id"]
LABEL_COLUMN = "attack_cat"
NORMAL_LABEL = "Normal"

REPORTS_PATH = Path("reports/metrics/unsw/two_stage/stage1")
FIGURES_PATH = Path("reports/figures/unsw/two_stage/stage1")
MODELS_PATH = Path("models/classification/two_stage/unsw/stage1")

for p in [REPORTS_PATH, FIGURES_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

#LOADING
print(f"\n{'='*60}")
print(f"    IDS Stage 1 -  Binary Classification (Normal vs Attack)")
print(f"\n{'='*60}\n")

train_df = pd.read_csv("data/processed/UNSW-NB15/splits/train.csv")
test_df = pd.read_csv("data/processed/UNSW-NB15/splits/test.csv")
val_df = pd.read_csv("data/processed/UNSW-NB15/splits/validation.csv")

y_train_raw = train_df[LABEL_COLUMN]
y_test_raw = test_df[LABEL_COLUMN]
y_val_raw = val_df[LABEL_COLUMN]

X_train = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])
X_val = val_df.drop(columns=[c for c in DROP_COLUMNS if c in val_df.columns])

#BINARY LABELS

y_train_bin = (y_train_raw != NORMAL_LABEL).astype(int).values
y_test_bin = (y_test_raw != NORMAL_LABEL).astype(int).values
y_val_bin = (y_val_raw != NORMAL_LABEL).astype(int).values

print("DISTRIBUTION - TRAIN")
for label, count in zip([0,1], [np.sum(y_train_bin==0), np.sum(y_train_bin==1)]):
    name = "Normal" if label == 0 else "Attack"
    pct = count / len(y_train_bin)*100
    print(f"    {label} ({name}): {count, } ({pct:.1}%)")

#PREPROCESSING

print("Preprocessing...")
preprocessor = joblib.load("models/preprocessing/binary/unsw/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_val_proc = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]

#WINDOWING
print("windowing...")

window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=False)

X_train_ae, X_train_seq, y_train_w = window_builder.transform(X_train_proc, y_train_bin)
X_test_ae, X_test_seq, y_test_w = window_builder.transform(X_test_proc, y_test_bin)
X_val_ae, X_val_seq, y_val_w = window_builder.transform(X_val_proc, y_val_bin)

print(f"    Shapes: train={X_train_seq.shape}   |   test={X_test_seq.shape}")

#BALANCED BATCHES

print(
    f"creating weighted train dataset "
    f"(balanced × normal_weight={NORMAL_WEIGHT}, attack_weight={ATTACK_WEIGHT})..."
)

train_dataset = create_balanced_tf_dataset(
    X_train_ae,
    X_train_seq,
    to_categorical(y_train_w, num_classes=2),
    batch_size=BATCH_SIZE,
    attack_weight=ATTACK_WEIGHT,
    normal_weight=NORMAL_WEIGHT,
)

#MODEL

print(f"building model...")

model = IDSModelFactory.create_model(window_size=WINDOW_SIZE, num_features=num_features, num_classes=2)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics={
        "classification": ["accuracy", Precision(name="precision"), Recall(name="recall"), F1Score(name="f1_score", average="macro"), AUC(name="auc")]
    },
    loss={
        "classification": BinaryFocalCrossentropy(gamma=3.0, alpha=0.25),
        "reconstruction": "mse",  
    },
    loss_weights={
        "classification": 1.0,
        "reconstruction": 0.05,
    }
)

model.summary()

#CALLBACKS

y_val_ohe = to_categorical(y_val_w, num_classes=2)

f1_callbacks = F1EarlyStopping(
    validation_data=(
        {
            "ae_input": X_val_ae,
            "cnn_input": X_val_seq,
            "lstm_input": X_val_seq,
        },
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_ae
        },
    ),
    patience=PATIENCE,
)

checkpoint = ModelCheckpoint(
    filepath=str(MODELS_PATH/"best_model_binary.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

#TRAINING

print(f"training model ({EPOCHS} epochs, {PATIENCE} patience)...")

steps_per_epoch = len(X_train_ae) // BATCH_SIZE

history = model.fit(
    train_dataset,
    validation_data=(
        {
            "ae_input": X_val_ae,
            "cnn_input": X_val_seq,
            "lstm_input": X_val_seq,
        },
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_ae,
        },
    ),
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint],
    # callbacks=[f1_callbacks, checkpoint, reduce_lr],
    verbose=1
)

#THRESHOLD

print("threshold calibration")

y_val_probs = model.predict(
    {
        "ae_input": X_val_ae,
        "cnn_input": X_val_seq,
        "lstm_input": X_val_seq,
    }
)["classification"][:, 1]

best_threshold = 0.5
best_f1 = 0.0

print(f"     {'Thresh':>6} | {'Attack Rec':>10} | {'Normal Rec':>10} | {'Macro F1':>8}")
print(f"     {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

for thresh in np.arange(0.05, 0.96, 0.05):
    y_pred = (y_val_probs > thresh).astype(int)
    report = classification_report(y_val_w, y_pred, output_dict=True, zero_division=0)
    attack_rec = report["1"]["recall"]
    normal_rec = report["0"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"     {thresh:>6.2f} | {attack_rec:>10.3f} | {normal_rec:>10.3f} | {macro_f1:>8.3f}")
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_threshold = thresh

print(f"\nselected threshold: {best_threshold:.2f} (best Macro F1: {best_f1:.3f})")

with open(MODELS_PATH / "threshold.json", "w") as f:
    json.dump({"threshold": float(best_threshold)}, f)

#EVALUATION

print("evaluating model...")

y_test_probs = model.predict(
    {
        "ae_input": X_test_ae,
        "cnn_input": X_test_seq,
        "lstm_input": X_test_seq,
    }
)["classification"][:, 1]

y_test_pred = (y_test_probs > best_threshold).astype(int)

report = classification_report(
    y_true=y_test_w,
    y_pred=y_test_pred,
    target_names=["Normal", "Attack"],
    zero_division=0,
    output_dict=True,
) 

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH / "classification_report.csv")

print(f"\nCLASSIFICATION REPORT:")
print(report_df.to_string())

print(f"\n{'='*60}")
print(f"\n  Stage 1 Results")
print(f"\n  Threshold: {best_threshold:.2f}")
print(f"  Normal Recall: {report['Normal']['recall']:.4f}")
print(f"  Attack Recall:  {report['Attack']['recall']:.4f}")
print(f"  Macro F1:       {report['macro avg']['f1-score']:.4f}")
print(f"\n{'='*60}")
