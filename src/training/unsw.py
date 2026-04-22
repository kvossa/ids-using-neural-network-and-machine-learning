"""
IDS — Entrenamiento UNSW multiclase
===================================
Estrategia combinada:
  - SMOTE selectivo para clases con < TARGET_MIN muestras
  - Focal loss con alpha ajustado inversamente a la frecuencia de clase
  - gamma configurable (default 3.0)

Uso:
    python -m src.training.train_unsw_multiclass
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from keras.metrics import AUC, Precision, Recall, F1Score
from keras.utils import to_categorical
from keras.losses import CategoricalFocalCrossentropy  
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.model import IDSModelFactory
from src.preprocessing.windowing import WindowGenerator
from src.utils.train_stopper import F1EarlyStopping

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

WINDOW_SIZE  = 10
WINDOW_STEP  = 1
EPOCHS       = 50
BATCH_SIZE   = 128
PATIENCE     = 15
FOCAL_GAMMA  = 2.0          # gamma más bajo → menos énfasis en hard samples
TARGET_MIN   = 12000        # clases con menos muestras se oversamplearán hasta aquí
LEARNING_RATE = 5e-4        # menor learning rate para mejor convergencia

REPORTS_PATH = Path("reports/metrics/unsw/multiclass/smote_focal")
FIGURES_PATH = Path("reports/figures/unsw/multiclass/smote_focal")
MODELS_PATH  = Path("models/classification/multiclass/unsw/smote_focal")

for p in [REPORTS_PATH, FIGURES_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================

print(f"\n{'='*60}")
print(f"  IDS UNSW MULTICLASE — SMOTE + Focal Loss Adaptativo")
print(f"{'='*60}\n")

train_df = pd.read_csv("data/processed/UNSW-NB15/splits/train.csv")
test_df  = pd.read_csv("data/processed/UNSW-NB15/splits/test.csv")
val_df   = pd.read_csv("data/processed/UNSW-NB15/splits/validation.csv")

DROP_COLS = ["attack_cat", "label", "id"]
LABEL_COL = "attack_cat"

y_train_raw = train_df[LABEL_COL]
y_test_raw  = test_df[LABEL_COL]
y_val_raw   = val_df[LABEL_COL]

X_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
X_test  = test_df.drop(columns=[c  for c in DROP_COLS if c in test_df.columns])
X_val   = val_df.drop(columns=[c   for c in DROP_COLS if c in val_df.columns])

# ==============================================================================
# 2. ENCODING DE ETIQUETAS
# ==============================================================================

label_encoder = joblib.load("models/preprocessing/multiclass/unsw/label_encoder.pkl")

y_train_enc = label_encoder.transform(y_train_raw)
y_test_enc  = label_encoder.transform(y_test_raw)
y_val_enc   = label_encoder.transform(y_val_raw)

NUM_CLASSES = len(label_encoder.classes_)
class_names = label_encoder.classes_.tolist()

print("[1] Distribución de clases — TRAIN:")
counts_orig = pd.Series(y_train_enc).value_counts().sort_index()
for cls_idx, cnt in counts_orig.items():
    flag = " ← oversampling" if cnt < TARGET_MIN else ""
    print(f"    {cls_idx} {class_names[cls_idx]:<20}: {cnt:>8,}{flag}")

# ==============================================================================
# 3. PREPROCESAMIENTO
# ==============================================================================

print("\n[2] Preprocesamiento...")
preprocessor = joblib.load("models/preprocessing/multiclass/unsw/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
X_val_proc   = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]
print(f"    Features: {num_features}")

# ==============================================================================
# 4. VENTANAS TEMPORALES
# ==============================================================================

print(f"\n[3] Ventanas temporales (size={WINDOW_SIZE}, step={WINDOW_STEP})...")

window_builder = WindowGenerator(
    window_size=WINDOW_SIZE,
    step=WINDOW_STEP,
    pure_windows_only=False,
)

X_train_ae, X_train_seq, y_train_w = window_builder.transform(X_train_proc, y_train_enc)
X_test_ae,  X_test_seq,  y_test_w  = window_builder.transform(X_test_proc,  y_test_enc)
X_val_ae,   X_val_seq,   y_val_w   = window_builder.transform(X_val_proc,   y_val_enc)

print(f"    Post-windowing — train: {X_train_seq.shape} | test: {X_test_seq.shape}")

# ==============================================================================
# 5. SMOTE SELECTIVO
# ==============================================================================

print(f"\n[4] SMOTE selectivo (target mínimo por clase: {TARGET_MIN:,})...")

counts_win = pd.Series(y_train_w).value_counts().sort_index()

# Solo oversamplear clases con menos de TARGET_MIN muestras
sampling_strategy = {
    int(cls_idx): TARGET_MIN
    for cls_idx, cnt in counts_win.items()
    if cnt < TARGET_MIN
}

# TARGET_PER_CLASS = {
#     class_names.index('Analysis')     : 8_000,
#     class_names.index('Backdoor')     : 8_000,
#     class_names.index('DoS')          : 15_000,  # ← más muestras para DoS
#     class_names.index('Shellcode')    : 8_000,
#     class_names.index('Worms')        : 5_000,   # límite por pocas muestras reales
# }

# sampling_strategy = {
#     cls_idx: target
#     for cls_idx, target in TARGET_PER_CLASS.items()
#     if counts_win.get(cls_idx, 0) < target
# }

if sampling_strategy:
    print("    Clases a oversamplear:")
    for cls_idx, target in sampling_strategy.items():
        print(f"      {class_names[cls_idx]:<20}: "
              f"{counts_win[cls_idx]:,} → {target:,}")

    # k_neighbors = mínimo entre 5 y (mínima clase - 1)
    k_nn = max(1, min(5, counts_win.min() - 1))
    print(f"    k_neighbors: {k_nn}")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_nn,
        random_state=42,
    )

    # SMOTE opera en 2D sobre X_ae
    X_ae_bal, y_bal = smote.fit_resample(X_train_ae, y_train_w)

    # Reconstruir X_seq:
    # - Muestras originales: preservar su ventana real
    # - Muestras sintéticas: repetir el frame sintético WINDOW_SIZE veces
    n_orig      = len(X_train_ae)
    X_syn_ae    = X_ae_bal[n_orig:]                              # (n_syn, features)
    X_seq_syn   = np.repeat(
        X_syn_ae.reshape(-1, 1, num_features),
        WINDOW_SIZE,
        axis=1,
    )                                                             # (n_syn, window, features)
    X_seq_bal   = np.vstack([X_train_seq, X_seq_syn])           # (n_orig + n_syn, window, features)

    print(f"\n    Post-SMOTE — train: {X_seq_bal.shape}")
    print("    Distribución post-SMOTE:")
    for cls_idx, cnt in pd.Series(y_bal).value_counts().sort_index().items():
        print(f"      {class_names[cls_idx]:<20}: {cnt:,}")
else:
    print("    Ninguna clase requiere oversampling.")
    X_ae_bal  = X_train_ae
    X_seq_bal = X_train_seq
    y_bal     = y_train_w

# ==============================================================================
# 6. ALPHA ADAPTATIVO POR CLASE (usar distribución ORIGINAL, no balanceada)
# ==============================================================================

print(f"\n[5] Calculando alpha adaptativo (gamma={FOCAL_GAMMA})...")

# IMPORTANTE: usar counts_orig (antes de SMOTE) para que alpha refleje
# la verdadera dificultad de cada clase
counts_orig = pd.Series(y_train_w).value_counts().sort_index()
total_orig = counts_orig.sum()

# Alpha inversamente proporcional a la frecuencia — clases raras reciben más peso
alpha_raw = 1.0 - (counts_orig / total_orig)
alpha_norm = (alpha_raw / alpha_raw.sum()).values.tolist()

print(f"    {'Clase':<22} {'n':>8} {'alpha':>8}")
print(f"    {'─'*42}")
for i, cls in enumerate(class_names):
    cls_idx = label_encoder.transform([cls])[0]
    n = counts_orig.get(cls_idx, 0)
    a = alpha_norm[i] if i < len(alpha_norm) else 0
    print(f"    {cls:<22} {n:>8,} {a:>8.4f}")

# ==============================================================================
# 7. PREPARAR INPUTS
# ==============================================================================

# Shuffle explícito del train balanceado
shuffle_idx = np.random.RandomState(42).permutation(len(X_ae_bal))
X_ae_bal    = X_ae_bal[shuffle_idx]
X_seq_bal   = X_seq_bal[shuffle_idx]
y_bal       = y_bal[shuffle_idx]

X_train_inputs = {
    "ae_input":   X_ae_bal,
    "cnn_input":  X_seq_bal,
    "lstm_input": X_seq_bal,
}
X_val_inputs = {
    "ae_input":   X_val_ae,
    "cnn_input":  X_val_seq,
    "lstm_input": X_val_seq,
}
X_test_inputs = {
    "ae_input":   X_test_ae,
    "cnn_input":  X_test_seq,
    "lstm_input": X_test_seq,
}

y_train_ohe = to_categorical(y_bal,      num_classes=NUM_CLASSES)
y_val_ohe   = to_categorical(y_val_w,    num_classes=NUM_CLASSES)
y_test_ohe  = to_categorical(y_test_w,   num_classes=NUM_CLASSES)

# Sample weights unitarios — el balanceo lo maneja el alpha de focal loss

# sw_train = np.ones(len(y_bal))

counts_orig = pd.Series(y_train_w).value_counts().sort_index()
max_count   = counts_orig.max() 

sw_per_class = {
    cls_idx: np.sqrt(max_count / cnt)
    for cls_idx, cnt in counts_orig.items()
}

MAX_WEIGHT = 10.0
sw_per_class = {k: min(v, MAX_WEIGHT) for k, v in sw_per_class.items()}

sw_train = np.array([sw_per_class[yi] for yi in y_bal])

print("Sample weights calculados (sqrt inverse freq, cap=5):")
for cls_idx, w in sorted(sw_per_class.items()):
    print(f"  {class_names[cls_idx]:<22}: {w:.3f}")

# Verificación de cardinalidad
assert X_ae_bal.shape[0]  == len(y_bal), "❌ Cardinalidad train ae"
assert X_seq_bal.shape[0] == len(y_bal), "❌ Cardinalidad train seq"
print("\n[6] ✅ Cardinalidad verificada")


# ==============================================================================
# 8. MODELO
# ==============================================================================

print("\n[7] Construyendo modelo...")

model = IDSModelFactory.create_model(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    num_classes=NUM_CLASSES,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics={
        "classification": [
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            F1Score(name="f1_score", average="macro"),
            AUC(name="auc", multi_label=True),
        ],
    },
    loss={
        "classification": CategoricalFocalCrossentropy(
            gamma=FOCAL_GAMMA,
            alpha=alpha_norm,
        ),
        "reconstruction": "mse",
    },
    loss_weights={
        "classification": 1.0,
        "reconstruction": 0.001,
    },
)

model.summary()

# ==============================================================================
# 9. CALLBACKS
# ==============================================================================

f1_cb = F1EarlyStopping(
    validation_data=(
        X_val_inputs,
        {"classification": y_val_ohe, "reconstruction": X_val_ae},
    ),
    patience=PATIENCE,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODELS_PATH / "best_model.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=0,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=0,
)

class EpochSummary(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"  Época {epoch+1:3d} | "
            f"loss: {logs.get('classification_loss', 0):.4f} | "
            f"f1: {logs.get('classification_f1_score', 0):.4f} | "
            f"val_f1: {logs.get('val_classification_f1_score', 0):.4f} | "
            f"lr: {float(self.model.optimizer.learning_rate):.6f}"
        )

# ==============================================================================
# 10. ENTRENAMIENTO
# ==============================================================================

print(f"\n[8] Entrenando ({EPOCHS} épocas máx, patience={PATIENCE})...")

history = model.fit(
    X_train_inputs,
    {
        "classification": y_train_ohe,
        "reconstruction": X_ae_bal,
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
    sample_weight={
        "classification": sw_train,
        "reconstruction": sw_train,
    },
    shuffle=True,
    callbacks=[f1_cb, checkpoint, reduce_lr, EpochSummary()],
    verbose=0,
)

# ==============================================================================
# 11. EVALUACIÓN
# ==============================================================================

print("\n[9] Evaluando sobre test set...")

y_pred_probs = model.predict(X_test_inputs, verbose=0)["classification"]
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = y_test_w

# ==============================================================================
# 12. MÉTRICAS
# ==============================================================================

print("\n[10] Guardando métricas...")

report = classification_report(
    y_true=y_true,
    y_pred=y_pred,
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH / "classification_report.csv")

history_df = pd.DataFrame(history.history)
history_df.to_csv(REPORTS_PATH / "training_metrics.csv")

# Guardar configuración del experimento
config_df = pd.DataFrame([{
    "window_size":  WINDOW_SIZE,
    "window_step":  WINDOW_STEP,
    "focal_gamma":  FOCAL_GAMMA,
    "target_min":   TARGET_MIN,
    "epochs_run":   len(history_df),
    "batch_size":   BATCH_SIZE,
}])
config_df.to_csv(REPORTS_PATH / "experiment_config.csv", index=False)

# ==============================================================================
# 13. VISUALIZACIONES
# ==============================================================================

print("\n[11] Generando visualizaciones...")

# Curvas de entrenamiento
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"UNSW Multiclase — SMOTE + Focal Loss (gamma={FOCAL_GAMMA})", fontsize=13)
for ax, (tm, vm, title) in zip(axes, [
    ("classification_loss",     "val_classification_loss",     "Loss"),
    ("classification_accuracy", "val_classification_accuracy", "Accuracy"),
    ("classification_f1_score", "val_classification_f1_score", "F1 Macro"),
]):
    if tm in history_df.columns:
        ax.plot(history_df[tm],  label="Train")
        ax.plot(history_df[vm],  label="Val", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.legend()
        ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_PATH / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names, ax=ax,
)
ax.set_title(f"Matriz de confusión — UNSW Multiclase (SMOTE + Focal)")
ax.set_ylabel("Real")
ax.set_xlabel("Predicho")
plt.tight_layout()
plt.savefig(FIGURES_PATH / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# F1 por clase — comparativa visual
f1_per_class = [report[cls]["f1-score"] for cls in class_names if cls in report]
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(class_names, f1_per_class, color=[
    "#2ecc71" if f >= 0.6 else "#f39c12" if f >= 0.3 else "#e74c3c"
    for f in f1_per_class
])
ax.set_xlim(0, 1)
ax.set_xlabel("F1-score")
ax.set_title("F1 por clase — UNSW Multiclase")
ax.axvline(x=0.3, color="gray", linestyle="--", alpha=0.5, label="F1=0.3")
ax.axvline(x=0.6, color="gray", linestyle=":",  alpha=0.5, label="F1=0.6")
ax.legend()
for bar, val in zip(bars, f1_per_class):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_PATH / "f1_per_class.png", dpi=150, bbox_inches="tight")
plt.close()

# ==============================================================================
# 14. RESUMEN FINAL
# ==============================================================================

macro = report.get("macro avg", {})
print(f"\n{'='*60}")
print(f"  RESUMEN FINAL — UNSW MULTICLASE (SMOTE + Focal Loss)")
print(f"{'='*60}")
print(f"  Gamma focal loss : {FOCAL_GAMMA}")
print(f"  Target SMOTE     : {TARGET_MIN:,} muestras mínimas")
print(f"  Épocas entrenadas: {len(history_df)}")
print(f"\n  Accuracy  : {report.get('accuracy', 0):.4f}")
print(f"  Macro F1  : {macro.get('f1-score', 0):.4f}")
print(f"  Macro P   : {macro.get('precision', 0):.4f}")
print(f"  Macro R   : {macro.get('recall', 0):.4f}")
print(f"\n  {'Clase':<22} {'F1':>8} {'Recall':>8} {'Precision':>10} {'Support':>10}")
print(f"  {'─'*62}")
for cls in class_names:
    if cls in report:
        r = report[cls]
        flag = ""
        if r["f1-score"] >= 0.6:
            flag = " ✅"
        elif r["f1-score"] >= 0.3:
            flag = " ⚠️"
        else:
            flag = " ❌"
        print(
            f"  {cls:<22} {r['f1-score']:>8.3f} "
            f"{r['recall']:>8.3f} {r['precision']:>10.3f} "
            f"{int(r['support']):>10,}{flag}"
        )
print(f"\n  Reportes  → {REPORTS_PATH}")
print(f"  Figuras   → {FIGURES_PATH}")
print(f"  Modelo    → {MODELS_PATH}")
print(f"{'='*60}\n")

# cm = confusion_matrix(y_true, y_pred)
# cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
# print(cm_df.loc[['DoS', 'Analysis', 'Backdoor']].to_string())

# pairs_problematic = [
#     ('DoS',      'Exploits'),
#     ('DoS',      'Fuzzers'),
#     ('Analysis', 'Fuzzers'),
#     ('Backdoor', 'Fuzzers'),
#     ('Shellcode','Reconnaissance'),
# ]

# print(f"{'Par':<35} {'Sep máx':>10} {'Sep media':>12}")
# print("-"*60)
# for a, b in pairs_problematic:
#     ia = label_encoder.transform([a])[0]
#     ib = label_encoder.transform([b])[0]
#     Xa = X_train_ae[y_train_w == ia]
#     Xb = X_train_ae[y_train_w == ib]
#     sep = np.abs(Xa.mean(0) - Xb.mean(0))
#     print(f"{a:<15} vs {b:<15} {sep.max():>10.4f} {sep.mean():>12.4f}")


# feature_names = preprocessor.pipeline.named_steps['feature_selection'].selected_features_

# print("\nFeatures más discriminativas para DoS vs Exploits:")
# ia = label_encoder.transform(['DoS'])[0]
# ib = label_encoder.transform(['Exploits'])[0]
# Xa = X_train_ae[y_train_w == ia]
# Xb = X_train_ae[y_train_w == ib]
# sep = np.abs(Xa.mean(0) - Xb.mean(0))
# sep_df = pd.DataFrame({'feature': feature_names, 'separacion': sep})
# print(sep_df.sort_values('separacion', ascending=False).head(10).to_string())