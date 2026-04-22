"""
IDS — Entrenamiento clasificación multiclase
=============================================
Ejecuta tres estrategias de balanceo en secuencia y guarda resultados
comparativos para elegir la mejor.

Estrategias:
  A) sample_weight    — pesos por clase, sin tocar los datos
  B) index_balancing  — RandomUnderSampler sobre índices post-windowing
  C) focal_loss_only  — focal loss sin ningún resampling

Configuración: edita solo el bloque CONFIG más abajo.
"""

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from keras.metrics import AUC, Precision, Recall, F1Score
from keras.utils import to_categorical
from keras.losses import CategoricalFocalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.model import IDSModelFactory
from src.preprocessing.windowing import WindowGenerator
from src.utils.train_stopper import F1EarlyStopping
from src.utils.visualize import IDSVisualizer

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

DATASET = "UNSW"   # "CIC" | "UNSW"

CONFIG = {
    "CIC": {
        "train_path":   Path("data/processed/CIC-IDS2017/splits/train/data.parquet"),
        "test_path":    Path("data/processed/CIC-IDS2017/splits/test/data.parquet"),
        "val_path":     Path("data/processed/CIC-IDS2017/splits/val/data.parquet"),
        "label_col":    "attack_type",
        "drop_cols":    ["Label", "attack_label", "attack_type", "source_file"],
        "sort_cols":    ["source_file", "Flow Duration"],
        "loader":       "parquet",
        "window_size":  5,
        "window_step":  1,
        "pure_windows": False,
        "shuffle_train": True,   # CIC necesita shuffle explícito
    },
    "UNSW": {
        "train_path":   Path("data/processed/UNSW-NB15/splits/train.csv"),
        "test_path":    Path("data/processed/UNSW-NB15/splits/test.csv"),
        "val_path":     Path("data/processed/UNSW-NB15/splits/validation.csv"),
        "label_col":    "attack_cat",
        "drop_cols":    ["attack_cat", "label", "id"],
        "sort_cols":    None,
        "loader":       "csv",
        "window_size":  10,
        "window_step":  1,
        "pure_windows": False,
        "shuffle_train": False,
    },
}

cfg          = CONFIG[DATASET]
WINDOW_SIZE  = cfg["window_size"]
WINDOW_STEP  = cfg["window_step"]
EPOCHS       = 30
BATCH_SIZE   = 64
PATIENCE     = 10

REPORTS_BASE = Path(f"reports/metrics/{DATASET.lower()}/multiclass")
FIGURES_BASE = Path(f"reports/figures/{DATASET.lower()}/multiclass")
REPORTS_BASE.mkdir(parents=True, exist_ok=True)
FIGURES_BASE.mkdir(parents=True, exist_ok=True)

# Estrategias a ejecutar — comenta las que no quieras correr
STRATEGIES = [
    # "sample_weight",
    # "index_balancing",
    "focal_loss_only",
]

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================

print(f"\n{'='*60}")
print(f"  IDS MULTICLASS — {DATASET}")
print(f"{'='*60}\n")

def load_df(path, loader):
    return pd.read_csv(path) if loader == "csv" else pd.read_parquet(path)

train_df = load_df(cfg["train_path"], cfg["loader"])
test_df  = load_df(cfg["test_path"],  cfg["loader"])
val_df   = load_df(cfg["val_path"],   cfg["loader"])

# Ordenamiento temporal proxy (CIC)
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
# 2. ENCODING DE ETIQUETAS
# ==============================================================================

label_encoder = joblib.load(f"models/preprocessing/multiclass/{DATASET.lower()}/label_encoder.pkl")

y_train_enc = label_encoder.transform(y_train_raw)
y_test_enc  = label_encoder.transform(y_test_raw)
y_val_enc   = label_encoder.transform(y_val_raw)

NUM_CLASSES = len(label_encoder.classes_)
class_names = label_encoder.classes_.tolist()

print(f"[1] Clases ({NUM_CLASSES}):")
counts = pd.Series(y_train_enc).value_counts().sort_index()
for i, (cls_idx, cnt) in enumerate(counts.items()):
    print(f"    {cls_idx} ({class_names[cls_idx]:<20}): {cnt:>8,}  ({cnt/len(y_train_enc)*100:.1f}%)")

# ==============================================================================
# 3. PREPROCESAMIENTO
# ==============================================================================

print(f"\n[2] Preprocesamiento...")
preprocessor = joblib.load(f"models/preprocessing/multiclass/{DATASET.lower()}/preprocessing.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
X_val_proc   = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]
print(f"    Features: {num_features}")
print(f"    Train: {X_train_proc.shape} | Test: {X_test_proc.shape} | Val: {X_val_proc.shape}")

# ==============================================================================
# 4. VENTANAS TEMPORALES
# ==============================================================================

print(f"\n[3] Ventanas temporales (size={WINDOW_SIZE}, step={WINDOW_STEP})...")

window_builder = WindowGenerator(
    window_size=WINDOW_SIZE,
    step=WINDOW_STEP,
    pure_windows_only=cfg["pure_windows"],
)

X_train_ae, X_train_seq, y_train_w = window_builder.transform(X_train_proc, y_train_enc)
X_test_ae,  X_test_seq,  y_test_w  = window_builder.transform(X_test_proc,  y_test_enc)
X_val_ae,   X_val_seq,   y_val_w   = window_builder.transform(X_val_proc,   y_val_enc)

# Shuffle explícito (CIC)
if cfg["shuffle_train"]:
    idx = np.random.RandomState(42).permutation(len(X_train_ae))
    X_train_ae  = X_train_ae[idx]
    X_train_seq = X_train_seq[idx]
    y_train_w   = y_train_w[idx]

print(f"    X_train_seq: {X_train_seq.shape}")
print(f"    X_val_seq  : {X_val_seq.shape}")
print(f"    X_test_seq : {X_test_seq.shape}")
print(f"\n    Distribución post-windowing:")
for cls_idx, cnt in pd.Series(y_train_w).value_counts().sort_index().items():
    print(f"      {class_names[cls_idx]:<22}: {cnt:>8,}")

# Inputs fijos para val y test (no se balancean)
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

y_val_ohe  = to_categorical(y_val_w,  num_classes=NUM_CLASSES)
y_test_ohe = to_categorical(y_test_w, num_classes=NUM_CLASSES)

# ==============================================================================
# 5. FUNCIONES AUXILIARES
# ==============================================================================

def build_and_compile(strategy: str, num_classes: int, num_features: int,
                      window_size: int, sample_weights=None):
    """Construye y compila el modelo según la estrategia de balanceo."""
    model = IDSModelFactory.create_model(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
    )

    if strategy == "focal_loss_only":
        clf_loss = CategoricalFocalCrossentropy(gamma=3.0, alpha=0.25)
    else:
        clf_loss = "categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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
            "classification": clf_loss,
            "reconstruction": "mse",
        },
        loss_weights={
            "classification": 1.0,
            "reconstruction": 0.001,
        },
    )
    return model


def prepare_train_inputs(strategy: str, X_ae, X_seq, y_w):
    """
    Prepara X_train_inputs, y_train_ohe y sample_weights
    según la estrategia de balanceo.
    """
    if strategy == "index_balancing":
        rus = RandomUnderSampler(random_state=42)
        idx_2d = np.arange(len(X_ae)).reshape(-1, 1)
        idx_bal, y_bal = rus.fit_resample(idx_2d, y_w)
        idx_bal = idx_bal.ravel()

        X_ae_use  = X_ae[idx_bal]
        X_seq_use = X_seq[idx_bal]
        y_use     = y_bal
        sw        = np.ones(len(y_bal))   # sin pesos diferenciales

        print(f"    Post-balanceo: {len(y_bal):,} muestras "
              f"({pd.Series(y_bal).value_counts().min():,} por clase)")

    else:
        X_ae_use  = X_ae
        X_seq_use = X_seq
        y_use     = y_w

        classes = np.unique(y_w)
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_w)
        cw_dict = dict(zip(classes, cw))

        if strategy == "sample_weight":
            sw = np.array([cw_dict[yi] for yi in y_use])
        else:  # focal_loss_only
            sw = np.ones(len(y_use))

    X_inputs = {
        "ae_input":   X_ae_use,
        "cnn_input":  X_seq_use,
        "lstm_input": X_seq_use,
    }
    y_ohe = to_categorical(y_use, num_classes=NUM_CLASSES)

    return X_inputs, y_ohe, sw, y_use


def save_results(strategy: str, history, y_true, y_pred, y_pred_probs):
    """Guarda métricas, reporte y visualizaciones de una estrategia."""
    reports_path = REPORTS_BASE / strategy
    figures_path = FIGURES_BASE / strategy
    reports_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    # Classification report
    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(reports_path / "classification_report.csv")

    # Training metrics
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(reports_path / "training_metrics.csv")

    # Curvas de entrenamiento
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{DATASET} multiclase — {strategy}", fontsize=13)
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
    plt.savefig(figures_path / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Matriz de confusión
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, NUM_CLASSES), max(5, NUM_CLASSES - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f"Confusión — {DATASET} ({strategy})")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(figures_path / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    return report


def print_summary(strategy: str, report: dict):
    macro = report.get("macro avg", {})
    print(f"\n  {'─'*50}")
    print(f"  Estrategia : {strategy}")
    print(f"  Accuracy   : {report.get('accuracy', 0):.4f}")
    print(f"  Macro F1   : {macro.get('f1-score', 0):.4f}")
    print(f"  Macro P    : {macro.get('precision', 0):.4f}")
    print(f"  Macro R    : {macro.get('recall', 0):.4f}")
    print(f"  {'─'*50}")
    # F1 por clase
    print(f"  {'Clase':<22} {'F1':>8} {'Recall':>8} {'Precision':>10}")
    for cls in class_names:
        if cls in report:
            r = report[cls]
            print(f"  {cls:<22} {r['f1-score']:>8.3f} {r['recall']:>8.3f} {r['precision']:>10.3f}")


# ==============================================================================
# 6. ENTRENAMIENTO POR ESTRATEGIA
# ==============================================================================

all_results = {}

for strategy in STRATEGIES:
    print(f"\n{'='*60}")
    print(f"  ESTRATEGIA: {strategy.upper()}")
    print(f"{'='*60}")

    # Preparar datos
    print(f"\n  Preparando datos...")
    X_train_inputs, y_train_ohe, sw, y_use = prepare_train_inputs(
        strategy, X_train_ae, X_train_seq, y_train_w
    )

    # Construir modelo
    model = build_and_compile(
        strategy=strategy,
        num_classes=NUM_CLASSES,
        num_features=num_features,
        window_size=WINDOW_SIZE,
    )

    # Callbacks
    ckpt_path = Path(f"models/classification/multiclass/{DATASET.lower()}/{strategy}/best_model.keras")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    f1_cb = F1EarlyStopping(
        validation_data=(
            X_val_inputs,
            {"classification": y_val_ohe, "reconstruction": X_val_ae},
        ),
        patience=PATIENCE,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
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

    # Entrenar
    print(f"\n  Entrenando ({EPOCHS} épocas máx, patience={PATIENCE})...")
    history = model.fit(
        X_train_inputs,
        {"classification": y_train_ohe, "reconstruction": X_train_inputs["ae_input"]},
        validation_data=(
            X_val_inputs,
            {"classification": y_val_ohe, "reconstruction": X_val_ae},
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight={
            "classification": sw,
            "reconstruction": np.ones(len(sw)),
        },
        shuffle=True,
        callbacks=[f1_cb, checkpoint, reduce_lr, EpochSummary()],
        verbose=0,
    )

    # Evaluar
    print(f"\n  Evaluando...")
    y_pred_probs = model.predict(X_test_inputs, verbose=0)["classification"]
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = y_test_w

    feature_names = preprocessor.pipeline.named_steps['feature_selection'].selected_features_
    X_arr = X_train_ae  

    # Separación entre pares de clases confundidas
    pairs = [
        ('DoS', 'Exploits'),
        ('DoS', 'Fuzzers'),
        ('Backdoor', 'Fuzzers'),
        ('Shellcode', 'Reconnaissance'),
        ('Fuzzers', 'Normal'),
    ]

    print(f"{'Par':<35} {'Sep máxima':>12} {'Sep media':>12}")
    print("-" * 62)
    for cls_a, cls_b in pairs:
        idx_a = label_encoder.transform([cls_a])[0]
        idx_b = label_encoder.transform([cls_b])[0]
        
        X_a = X_arr[y_train_w == idx_a]
        X_b = X_arr[y_train_w == idx_b]
        
        sep = np.abs(X_a.mean(axis=0) - X_b.mean(axis=0))
        print(f"{cls_a:<15} vs {cls_b:<15} {sep.max():>12.4f} {sep.mean():>12.4f}")

    # También ver cuántas muestras tiene cada clase en train
    print("\nMuestras por clase en train:")
    for cls in ['DoS', 'Backdoor', 'Shellcode', 'Worms', 'Fuzzers', 'Exploits', 'Normal']:
        idx = label_encoder.transform([cls])[0]
        n = (y_train_w == idx).sum()
        print(f"  {cls:<20}: {n:>8,}")

    report = save_results(strategy, history, y_true, y_pred, y_pred_probs)
    all_results[strategy] = report
    print_summary(strategy, report)



# ==============================================================================
# 7. TABLA COMPARATIVA FINAL
# ==============================================================================

print(f"\n\n{'='*60}")
print(f"  COMPARATIVA FINAL — {DATASET} MULTICLASE")
print(f"{'='*60}")
print(f"\n  {'Estrategia':<20} {'Accuracy':>10} {'Macro F1':>10} {'Macro R':>10}")
print(f"  {'─'*52}")

comparison = []
for strategy, report in all_results.items():
    macro = report.get("macro avg", {})
    acc   = report.get("accuracy", 0)
    f1    = macro.get("f1-score", 0)
    rec   = macro.get("recall", 0)
    comparison.append({"strategy": strategy, "accuracy": acc, "macro_f1": f1, "macro_recall": rec})
    print(f"  {strategy:<20} {acc:>10.4f} {f1:>10.4f} {rec:>10.4f}")

# Guardar comparativa
comp_df = pd.DataFrame(comparison)
comp_df.to_csv(REPORTS_BASE / "strategy_comparison.csv", index=False)

best = comp_df.loc[comp_df["macro_f1"].idxmax(), "strategy"]
print(f"\n  → Mejor estrategia por Macro F1: {best}")
print(f"\n  Reportes guardados en: {REPORTS_BASE}")
print(f"  Figuras  guardadas en: {FIGURES_BASE}")