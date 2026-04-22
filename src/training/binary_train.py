"""
IDS - Entrenamiento clasificación binaria (Normal vs Ataque)
============================================================
Objetivo: baseline limpio para validar que el modelo aprende antes
          de pasar a multiclase y estrategias de balanceo avanzadas.

Configuración por dataset:
  CIC-IDS2017  → clase normal = 'BENIGN'
  UNSW-NB15    → clase normal = 'Normal'
"""

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from keras.metrics import AUC, Precision, Recall, F1Score
from keras.utils import to_categorical
from keras.losses import BinaryFocalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.model import IDSModelFactory
from src.preprocessing.windowing import WindowGenerator
from src.utils.train_stopper import F1EarlyStopping
from src.utils.epoch_summary import EpochSummary
from src.utils.visualize import IDSVisualizer
# from src.training.focal_loss import focal_loss


# ==============================================================================
# CONFIGURACIÓN — edita solo este bloque para cambiar de dataset
# ==============================================================================

DATASET = "UNSW"   # "CIC" | "UNSW"

CONFIG = {
    "CIC": {
        "train_path":     Path("data/processed/CIC-IDS2017/splits/train/data.parquet"),
        "test_path":      Path("data/processed/CIC-IDS2017/splits/test/data.parquet"),
        "val_path":       Path("data/processed/CIC-IDS2017/splits/val/data.parquet"),
        "label_col":      "attack_type",
        "normal_label":   "BENIGN",
        "drop_cols":      ["Label", "attack_label", "attack_type", "source_file"],
        "sort_cols":      ["source_file", "Flow Duration"],   # orden temporal proxy
        "loader":         "parquet",
        "window_size":    1,    # CIC es features agregadas → window_size=1
        "window_step":    1,
        "pure_windows":   False,
    },
    "UNSW": {
        "train_path":     Path("data/processed/UNSW-NB15/splits/train.csv"),
        "test_path":      Path("data/processed/UNSW-NB15/splits/test.csv"),
        "val_path":       Path("data/processed/UNSW-NB15/splits/validation.csv"),
        "label_col":      "attack_cat",
        "normal_label":   "Normal",
        "drop_cols":      ["attack_cat", "label, ""id"],
        "sort_cols":      None,
        "loader":         "csv",
        "window_size":    10,
        "window_step":    1,
        "pure_windows":   False,
    },
}

cfg            = CONFIG[DATASET]
WINDOW_SIZE    = cfg["window_size"]
WINDOW_STEP    = cfg["window_step"]
EPOCHS         = 30
BATCH_SIZE     = 256
PATIENCE       = 10
REPORTS_PATH   = Path(f"reports/metrics/{DATASET.lower()}/binary")
FIGURES_PATH   = Path(f"reports/figures/{DATASET.lower()}/binary")
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================

print(f"\n{'='*60}")
print(f"  IDS BINARY CLASSIFIER — {DATASET}")
print(f"{'='*60}\n")

def load_df(path: Path, loader: str) -> pd.DataFrame:
    return pd.read_parquet(path) if loader == "parquet" else pd.read_csv(path)

train_df = load_df(cfg["train_path"], cfg["loader"])
test_df  = load_df(cfg["test_path"],  cfg["loader"])
val_df   = load_df(cfg["val_path"],   cfg["loader"])

feature_cols = [c for c in train_df.columns if c not in ['attack_cat', 'label', 'id']]
merged = train_df[feature_cols].merge(test_df[feature_cols], how='inner')
print(f"Filas idénticas entre train y test: {len(merged):,}")

# Ordenamiento temporal proxy (solo CIC)
if cfg["sort_cols"]:
    sort_cols = [c for c in cfg["sort_cols"] if c in train_df.columns]
    train_df  = train_df.sort_values(by=sort_cols).reset_index(drop=True)

label_col = cfg["label_col"]
drop_cols  = [c for c in cfg["drop_cols"] if c in train_df.columns]

y_train_raw = train_df[label_col]
y_test_raw  = test_df[label_col]
y_val_raw   = val_df[label_col]

X_train = train_df.drop(columns=drop_cols)
X_test  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
X_val   = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])

# ==============================================================================
# 2. BINARIZACIÓN DE ETIQUETAS
#    0 = Normal/BENIGN  |  1 = Ataque
# ==============================================================================

normal = cfg["normal_label"]
y_train_bin = (y_train_raw != normal).astype(int).values
y_test_bin  = (y_test_raw  != normal).astype(int).values
y_val_bin   = (y_val_raw   != normal).astype(int).values

print("[1] Distribución binaria — TRAIN")
counts = pd.Series(y_train_bin).value_counts().sort_index()
for label, count in counts.items():
    name = "Normal" if label == 0 else "Ataque"
    pct  = count / len(y_train_bin) * 100
    print(f"    {label} ({name}): {count:,}  ({pct:.1f}%)")

# ==============================================================================
# 3. PREPROCESAMIENTO
# ==============================================================================

print("\n[2] Aplicando pipeline de preprocesamiento...")


preprocessor = joblib.load(f"models/preprocessing/binary/{DATASET.lower()}/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
X_val_proc   = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]
print(f"    Features tras preprocesamiento: {num_features}")
print(f"    Shapes — train: {X_train_proc.shape} | test: {X_test_proc.shape} | val: {X_val_proc.shape}")

# selector = preprocessor.pipeline.named_steps['feature_selection']
# print("Features seleccionadas:")
# print(selector.selected_features_)

# ==============================================================================
# 4. VENTANAS TEMPORALES
# ==============================================================================

print(f"\n[3] Generando ventanas temporales (size={WINDOW_SIZE}, step={WINDOW_STEP})...")

window_builder = WindowGenerator(
    window_size=WINDOW_SIZE,
    step=WINDOW_STEP,
    pure_windows_only=cfg["pure_windows"],
)

X_train_ae, X_train_seq, y_train_w = window_builder.transform(X_train_proc, y_train_bin)
X_test_ae,  X_test_seq,  y_test_w  = window_builder.transform(X_test_proc,  y_test_bin)
X_val_ae,   X_val_seq,   y_val_w   = window_builder.transform(X_val_proc,   y_val_bin)

print(f"    X_train_seq : {X_train_seq.shape}")
print(f"    X_test_seq  : {X_test_seq.shape}")
print(f"    X_val_seq   : {X_val_seq.shape}")

shuffle_idx = np.random.RandomState(42).permutation(len(X_train_ae))
X_train_ae  = X_train_ae[shuffle_idx]
X_train_seq = X_train_seq[shuffle_idx]
y_train_w   = y_train_w[shuffle_idx]

# ==============================================================================
# 5. CLASS WEIGHTS  (sin resampling — más limpio para baseline)
# ==============================================================================

print("\n[4] Calculando class weights...")

cw = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train_w,
)
class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}

# cw = {0: 1.0, 1: 8.0}

print(f"    Class weights: {class_weight_dict}")

sample_weights = np.where(y_train_w == 1, cw[1], cw[0])

print(f"    Sample weights — Normal: {cw[0]:.4f} | Ataque: {cw[1]:.4f}")
print(f"    Shape sample_weights: {sample_weights.shape}")

# ==============================================================================
# 6. PREPARAR INPUTS AL MODELO
# ==============================================================================

NUM_CLASSES = 2

y_train_ohe = to_categorical(y_train_w, num_classes=NUM_CLASSES)
y_test_ohe  = to_categorical(y_test_w,  num_classes=NUM_CLASSES)
y_val_ohe   = to_categorical(y_val_w,   num_classes=NUM_CLASSES)

X_train_inputs = {
    "ae_input":   X_train_ae,
    "cnn_input":  X_train_seq,
    "lstm_input": X_train_seq,
}
X_test_inputs = {
    "ae_input":   X_test_ae,
    "cnn_input":  X_test_seq,
    "lstm_input": X_test_seq,
}
X_val_inputs = {
    "ae_input":   X_val_ae,
    "cnn_input":  X_val_seq,
    "lstm_input": X_val_seq,
}

# Verificación de cardinalidad
assert X_train_ae.shape[0] == len(y_train_w),  "❌ Cardinalidad train incorrecta"
assert X_test_ae.shape[0]  == len(y_test_w),   "❌ Cardinalidad test incorrecta"
assert X_val_ae.shape[0]   == len(y_val_w),    "❌ Cardinalidad val incorrecta"
print("\n[5] ✅ Cardinalidad verificada")

# ==============================================================================
# 7. MODELO
# ==============================================================================

print("\n[6] Construyendo y compilando modelo...")

model = IDSModelFactory.create_model(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    num_classes=NUM_CLASSES,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics={
        "classification": [
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            F1Score(name="f1_score", average="macro"),
            AUC(name="auc"),
        ],
    },
    loss={
        "classification": BinaryFocalCrossentropy(gamma=3.0, alpha=0.25),
        "reconstruction": "mse",
    },
    loss_weights={
        "classification": 1.0,
        "reconstruction": 0.05,
    },
)

model.summary()

# ==============================================================================
# 8. CALLBACKS
# ==============================================================================

f1_callback = F1EarlyStopping(
    validation_data=(
        X_val_inputs,
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_ae,
        },
    ),
    patience=PATIENCE,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"models/classification/binary/{DATASET.lower()}/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)
Path(f"models/binary/{DATASET.lower()}").mkdir(parents=True, exist_ok=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)



# ==============================================================================
# 9. ENTRENAMIENTO
# ==============================================================================

print("\n[7] Entrenando modelo...")


history = model.fit(
    X_train_inputs,
    {
        "classification": y_train_ohe,
        "reconstruction": X_train_ae,
    },
    validation_data=(
        X_val_inputs,
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_ae,
        },
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    # class_weight=class_weight_dict,
    # sample_weight={
    #     "classification": sample_weights,
    #     "reconstruction": np.ones(len(sample_weights)),  # reconstrucción sin peso diferencial
    # },
    shuffle=True,
    callbacks=[f1_callback, checkpoint, reduce_lr, EpochSummary()],
    verbose=1,
)


# ==============================================================================
# 10. EVALUACIÓN
# ==============================================================================

print("\n[8] Evaluando sobre test set...")

model.evaluate(
    X_test_inputs,
    {
        "classification": y_test_ohe,
        "reconstruction": X_test_ae,
    },
    verbose=1,
)

y_pred_probs = model.predict(X_test_inputs)["classification"]
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = y_test_w

# ¿Qué está prediciendo realmente?
print("Distribución de predicciones:")
print(pd.Series(y_pred).value_counts())

print("\nDistribución real:")
print(pd.Series(y_test_w).value_counts())

# ¿Qué tan seguros están las probabilidades?
print(f"\nConfianza media clase 0: {y_pred_probs[:,0].mean():.4f}")
print(f"Confianza media clase 1: {y_pred_probs[:,1].mean():.4f}")
print(f"Probabilidad mínima máxima: {y_pred_probs.max(axis=1).min():.4f}")

# ==============================================================================
# 11. MÉTRICAS Y REPORTES
# ==============================================================================

print("\n[9] Guardando métricas...")

class_names = ["Normal", "Ataque"]

report = classification_report(
    y_true=y_true,
    y_pred=y_pred,
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH / "classification_report.csv")
print("\nCLASSIFICATION REPORT:")
print(report_df.to_string())

history_df = pd.DataFrame(history.history)
history_df.to_csv(REPORTS_PATH / "training_metrics.csv")

# ==============================================================================
# 12. VISUALIZACIONES
# ==============================================================================

print("\n[10] Generando visualizaciones...")

# --- Curvas de entrenamiento ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Entrenamiento binario — {DATASET}", fontsize=13)

metrics_to_plot = [
    ("classification_loss",    "val_classification_loss",    "Loss"),
    ("classification_accuracy","val_classification_accuracy","Accuracy"),
    ("classification_f1_score","val_classification_f1_score","F1 Score"),
]
for ax, (train_m, val_m, title) in zip(axes, metrics_to_plot):
    if train_m in history_df.columns:
        ax.plot(history_df[train_m],  label="Train")
        ax.plot(history_df[val_m],    label="Val", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Matriz de confusión ---
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names, ax=ax
)
ax.set_title(f"Matriz de confusión — {DATASET} (binario)")
ax.set_ylabel("Real")
ax.set_xlabel("Predicho")
plt.tight_layout()
plt.savefig(FIGURES_PATH / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Resumen final en consola ---
print(f"\n{'='*60}")
print(f"  RESUMEN FINAL — {DATASET} (binario)")
print(f"{'='*60}")
print(f"  Accuracy : {report['accuracy']:.4f}")
print(f"  Precision (Ataque): {report['Ataque']['precision']:.4f}")
print(f"  Recall    (Ataque): {report['Ataque']['recall']:.4f}")
print(f"  F1        (Ataque): {report['Ataque']['f1-score']:.4f}")
print(f"  F1        (Normal): {report['Normal']['f1-score']:.4f}")
print(f"  Macro F1          : {report['macro avg']['f1-score']:.4f}")
print(f"\n  Reportes  → {REPORTS_PATH}")
print(f"  Figuras   → {FIGURES_PATH}")
print(f"{'='*60}\n")



