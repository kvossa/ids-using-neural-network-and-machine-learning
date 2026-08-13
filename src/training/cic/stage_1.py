import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import random

from keras.metrics import AUC, Precision, Recall, F1Score
from keras.utils import to_categorical
from keras.losses import  BinaryFocalCrossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.model.model import IDSModelFactory
from src.preprocessing.windowing.windowing import WindowGenerator
from src.utils.batch_balancer import create_balanced_tf_dataset
from src.utils.stage1_binary_scoring import CALIBRATOR_FILENAME
from src.config import CIC_STAGE1_DIR, DATA_PATHS, PREPROC_PATHS, REPORT_PATHS
import tensorflow as tf

#CONF


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return int(val)


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return float(val)


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return str(val)

WINDOW_SIZE = _env_int("IDS_STAGE1_WINDOW_SIZE", 5)
WINDOW_STEP = _env_int("IDS_STAGE1_WINDOW_STEP", 1)
EPOCHS = _env_int("IDS_STAGE1_EPOCHS", 30)
BATCH_SIZE = _env_int("IDS_STAGE1_BATCH", 256)
PATIENCE = _env_int("IDS_STAGE1_PATIENCE", 10)
# Sample weights: BENIGN (class 0) is the minority in CIC — do not overweight attack.
ATTACK_WEIGHT = _env_float("IDS_STAGE1_ATTACK_WEIGHT", 10.0)
NORMAL_WEIGHT = _env_float("IDS_STAGE1_NORMAL_WEIGHT", 5.0)
LEARNING_RATE = _env_float("IDS_STAGE1_LR", 1e-4)
 
# Threshold on validation: enforce dual constraints first.
MIN_NORMAL_RECALL_TARGET = _env_float("IDS_STAGE1_MIN_NORMAL_RECALL", 0.4)
MIN_ATTACK_RECALL_TARGET = _env_float("IDS_STAGE1_MIN_ATTACK_RECALL", 0.80)
MIN_ATTACK_RECALL_FLOOR = _env_float("IDS_STAGE1_MIN_ATTACK_FLOOR", 0.75)
USE_ISOTONIC_CALIBRATION = True
THRESHOLD_GRID_START = 0.02
THRESHOLD_GRID_END = 0.98
THRESHOLD_GRID_STEP = 0.01
TARGET_MACRO_F1 = 0.75
# "constrained": dual recall if possible, else Normal floor, else fallbacks.
# "max_val_macro_f1": pick t that maximizes val macro F1 (current ceiling reference).
THRESHOLD_SELECTION_MODE = _env_str("IDS_STAGE1_THRESHOLD_MODE", "constrained")

# Classification loss (binary focal)
FOCAL_GAMMA = _env_float("IDS_STAGE1_FOCAL_GAMMA", 2.0)
FOCAL_ALPHA = _env_float("IDS_STAGE1_FOCAL_ALPHA", 0.35)

# Extra train cycles: mine FP/FN on train (vs provisional val threshold), append copies, re-fit.
# Ceiling baseline uses no refinement loops (hn0).
# HARD_NEGATIVE_REFINEMENT_LOOPS = _env_int("IDS_STAGE1_HN_LOOPS", 0)
# HARD_NEGATIVE_EPOCHS = _env_int("IDS_STAGE1_HN_EPOCHS", 10)
# HARD_NEGATIVE_MAX_PER_CLASS = _env_int("IDS_STAGE1_HN_MAX_PER_CLASS", 50_000)
# HARD_NEGATIVE_DUPLICATES_FP = _env_int("IDS_STAGE1_HN_DUP_FP", 1)
# HARD_NEGATIVE_DUPLICATES_FN = _env_int("IDS_STAGE1_HN_DUP_FN", 1)
# Optional domain-aware boost for benign samples from a problematic source.
BENIGN_FOCUS_SOURCE = _env_str("IDS_STAGE1_BENIGN_FOCUS_SOURCE", "")
BENIGN_FOCUS_MULT = _env_float("IDS_STAGE1_BENIGN_FOCUS_MULT", 1.0)
BENIGN_FOCUS_MATCH = _env_str("IDS_STAGE1_BENIGN_FOCUS_MATCH", "exact")
RANDOM_SEED = _env_int("IDS_STAGE1_SEED", 42)
TRAIN_SUBSAMPLE = _env_float("IDS_TRAIN_SUBSAMPLE", 1.0)

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def _window_last(values: np.ndarray, window_size: int, step: int) -> np.ndarray:
    v = np.asarray(values)
    n = len(v)
    if n < window_size:
        raise ValueError("window_size is larger than available samples")
    return np.array([v[i + window_size - 1] for i in range(0, n - window_size + 1, step)])


def build_benign_focus_weights(y_int: np.ndarray, source_labels: np.ndarray):
    y = np.asarray(y_int).astype(np.int32).ravel()
    src = np.asarray(source_labels).astype(str)
    w = np.ones(len(y), dtype=np.float32)
    stats = {
        "enabled": False,
        "source": BENIGN_FOCUS_SOURCE,
        "match_mode": BENIGN_FOCUS_MATCH,
        "multiplier": float(BENIGN_FOCUS_MULT),
        "eligible_count": 0,
        "boosted_count": 0,
    }
    if BENIGN_FOCUS_SOURCE == "" or BENIGN_FOCUS_MULT <= 1.0:
        return w, stats

    benign_mask = y == 0
    if BENIGN_FOCUS_MATCH == "contains":
        src_mask = np.char.find(src, BENIGN_FOCUS_SOURCE) >= 0
    else:
        src_mask = src == BENIGN_FOCUS_SOURCE
    boosted = benign_mask & src_mask
    w[boosted] = float(BENIGN_FOCUS_MULT)

    stats["enabled"] = True
    stats["eligible_count"] = int(np.sum(benign_mask))
    stats["boosted_count"] = int(np.sum(boosted))
    return w, stats


def select_threshold(
    y_true,
    scores,
    *,
    mode: str,
    min_normal_recall: float,
    min_attack_recall_target: float,
    min_attack_recall_floor: float,
    grid_start: float,
    grid_end: float,
    grid_step: float,
):
    grid = np.arange(grid_start, grid_end + grid_step * 0.5, grid_step)
    rows = []
    for t in grid:
        y_pred = (scores > t).astype(int)
        r = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        n_rec = r["0"]["recall"]
        a_rec = r["1"]["recall"]
        macro_f1 = r["macro avg"]["f1-score"]
        bal = 0.5 * (n_rec + a_rec)
        rows.append((float(t), n_rec, a_rec, macro_f1, bal))

    best_any_macro = max(rows, key=lambda x: (x[3], x[4], -abs(x[0] - 0.5)))
    ceiling_meta = {
        "val_macro_f1_max": best_any_macro[3],
        "val_balanced_accuracy_at_macro_max": best_any_macro[4],
        "threshold_at_val_macro_f1_max": best_any_macro[0],
        "val_normal_recall_at_macro_max": best_any_macro[1],
        "val_attack_recall_at_macro_max": best_any_macro[2],
    }

    if mode == "max_val_macro_f1":
        best = best_any_macro
        meta = {
            "mode": "max_val_macro_f1",
            "val_normal_recall": best[1],
            "val_attack_recall": best[2],
            "val_macro_f1": best[3],
            "val_balanced_accuracy": best[4],
            "val_macro_f1_ceiling": ceiling_meta,
        }
        return best[0], meta

    feasible_dual = [
        row for row in rows if row[1] >= min_normal_recall and row[2] >= min_attack_recall_target
    ]
    if feasible_dual:
        best = max(feasible_dual, key=lambda x: (x[3], x[4]))
        meta = {
            "mode": "val_dual_recall_target",
            "val_normal_recall": best[1],
            "val_attack_recall": best[2],
            "val_macro_f1": best[3],
            "val_balanced_accuracy": best[4],
            "val_macro_f1_ceiling": ceiling_meta,
            "dual_feasible_count": len(feasible_dual),
        }
        return best[0], meta

    meta_prefix = {
        "val_macro_f1_ceiling": ceiling_meta,
        "dual_feasible_count": 0,
        "dual_constraint_note": (
            f"no threshold with Normal>={min_normal_recall:.2f} and "
            f"Attack>={min_attack_recall_target:.2f} on this val split"
        ),
    }

    feasible = [row for row in rows if row[1] >= min_normal_recall]
    if feasible:
        best = max(feasible, key=lambda x: (x[2], x[3]))
        meta = {
            "mode": "val_normal_recall_ge_target",
            "val_normal_recall": best[1],
            "val_attack_recall": best[2],
            "val_macro_f1": best[3],
            "val_balanced_accuracy": best[4],
            **meta_prefix,
        }
        return best[0], meta

    floor_ok = [row for row in rows if row[2] >= min_attack_recall_floor]
    if floor_ok:
        best = max(floor_ok, key=lambda x: (x[1], x[3]))
        meta = {
            "mode": "fallback_max_normal_with_attack_floor",
            "val_normal_recall": best[1],
            "val_attack_recall": best[2],
            "val_macro_f1": best[3],
            "val_balanced_accuracy": best[4],
            **meta_prefix,
        }
        return best[0], meta

    best = max(rows, key=lambda x: x[4])
    meta = {
        "mode": "fallback_balanced_accuracy",
        "val_normal_recall": best[1],
        "val_attack_recall": best[2],
        "val_macro_f1": best[3],
        "val_balanced_accuracy": best[4],
        **meta_prefix,
    }
    return best[0], meta


def evaluate_split(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)
    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0,
        output_dict=True,
    )
    return y_pred, report


def _subsample_indices(idxs: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if idxs.size == 0:
        return idxs.astype(np.int64)
    if idxs.size > max_n:
        return rng.choice(idxs, size=max_n, replace=False)
    return idxs.astype(np.int64)


def fit_iso_and_scores(model, X_val_ae, X_val_seq, y_val_w):
    y_val_raw = model.predict(
        {"ae_input": X_val_ae, "cnn_input": X_val_seq, "lstm_input": X_val_seq},
        verbose=0,
    )["classification"][:, 1]
    if not USE_ISOTONIC_CALIBRATION:
        return None, y_val_raw, y_val_raw
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_val_raw, y_val_w)
    return iso, y_val_raw, iso.predict(y_val_raw)


def mine_hard_negatives(
    model,
    X_train_ae,
    X_train_seq,
    y_train_w,
    source_train_w,
    iso,
    threshold: float,
    rng: np.random.Generator,
):
    y_tr_raw = model.predict(
        {"ae_input": X_train_ae, "cnn_input": X_train_seq, "lstm_input": X_train_seq},
        verbose=0,
    )["classification"][:, 1]
    if iso is not None:
        y_tr_s = iso.predict(y_tr_raw)
    else:
        y_tr_s = y_tr_raw
    pred = (y_tr_s > threshold).astype(int)
    fp_mask = (y_train_w == 0) & (pred == 1)
    fn_mask = (y_train_w == 1) & (pred == 0)
    fp_idx = np.flatnonzero(fp_mask)
    fn_idx = np.flatnonzero(fn_mask)
    fp_take = _subsample_indices(fp_idx, HARD_NEGATIVE_MAX_PER_CLASS, rng)
    fn_take = _subsample_indices(fn_idx, HARD_NEGATIVE_MAX_PER_CLASS, rng)
    parts_ae = [X_train_ae]
    parts_seq = [X_train_seq]
    parts_y = [y_train_w.astype(np.int32, copy=False)]
    parts_src = [np.asarray(source_train_w).astype(object)]
    for _ in range(HARD_NEGATIVE_DUPLICATES_FP):
        if fp_take.size:
            parts_ae.append(X_train_ae[fp_take])
            parts_seq.append(X_train_seq[fp_take])
            parts_y.append(np.zeros(fp_take.size, dtype=np.int32))
            parts_src.append(np.asarray(source_train_w)[fp_take])
    for _ in range(HARD_NEGATIVE_DUPLICATES_FN):
        if fn_take.size:
            parts_ae.append(X_train_ae[fn_take])
            parts_seq.append(X_train_seq[fn_take])
            parts_y.append(np.ones(fn_take.size, dtype=np.int32))
            parts_src.append(np.asarray(source_train_w)[fn_take])
    X_ae = np.concatenate(parts_ae, axis=0)
    X_seq = np.concatenate(parts_seq, axis=0)
    y_int = np.concatenate(parts_y, axis=0)
    src_out = np.concatenate(parts_src, axis=0)
    perm = rng.permutation(len(y_int))
    return (
        X_ae[perm],
        X_seq[perm],
        y_int[perm],
        src_out[perm],
        {"fp_train": int(fp_mask.sum()), "fn_train": int(fn_mask.sum()), "fp_used": int(fp_take.size), "fn_used": int(fn_take.size)},
    )


def compile_model(m):
    m.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics={
            "classification": [
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                F1Score(name="f1_score", average="macro"),
                AUC(name="auc"),
            ]
        },
        loss={
            "classification": BinaryFocalCrossentropy(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
            "reconstruction": "mse",
        },
        loss_weights={"classification": 1.0, "reconstruction": 0.05},
    )


def build_model_compile(n_feat: int):
    m = IDSModelFactory.create_model(window_size=WINDOW_SIZE, num_features=n_feat, num_classes=2)
    compile_model(m)
    return m


DROP_COLUMNS = ["Label", "attack_label", "attack_type", "source_file"]
LABEL_COLUMN = "attack_type"
NORMAL_LABEL = "BENIGN"

REPORTS_PATH = Path(REPORT_PATHS["cic_stage1"])
MODELS_PATH = Path(CIC_STAGE1_DIR)

for p in [REPORTS_PATH, MODELS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

#LOADING
print(f"\n{'='*60}")
print(f"    IDS Stage 1 -  Binary Classification (Normal vs Attack)")
print(f"\n{'='*60}\n")

train_df = pd.read_parquet(DATA_PATHS["cic"]["train"])
test_df = pd.read_parquet(DATA_PATHS["cic"]["test"])
val_df = pd.read_parquet(DATA_PATHS["cic"]["val"])

if TRAIN_SUBSAMPLE < 1.0:
    train_df = train_df.sample(frac=TRAIN_SUBSAMPLE, random_state=RANDOM_SEED)
    print(f"Subsampled train to {len(train_df)} rows ({TRAIN_SUBSAMPLE:.0%})")

#BINARY LABELS — extract as standalone numpy to free DataFrames

y_train_bin = (train_df[LABEL_COLUMN] != NORMAL_LABEL).astype(int).values
y_test_bin = (test_df[LABEL_COLUMN] != NORMAL_LABEL).astype(int).values
y_val_bin = (val_df[LABEL_COLUMN] != NORMAL_LABEL).astype(int).values

train_source = (
    train_df["source_file"].astype(str).to_numpy()
    if "source_file" in train_df.columns
    else np.array(["unknown"] * len(train_df), dtype=object)
)

# Drop columns from DataFrames
X_train = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])
X_val = val_df.drop(columns=[c for c in DROP_COLUMNS if c in val_df.columns])

# Free original DataFrames — no longer needed
del train_df, test_df, val_df
import gc; gc.collect()

counts = np.bincount(y_train_bin, minlength=2)
print("DISTRIBUTION - TRAIN")
for label, count in [(0, counts[0]), (1, counts[1])]:
    name = "Normal" if label == 0 else "Attack"
    pct = count / len(y_train_bin)*100
    print(f"    {label} ({name}): {count, } ({pct:.1}%)")

#PREPROCESSING

print("Preprocessing...")
preprocessor = joblib.load(PREPROC_PATHS["cic"]["binary_preprocessor"])

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_val_proc = preprocessor.transform(X_val)

num_features = X_train_proc.shape[1]

# Free raw DataFrames — no longer needed after preprocessing
del X_train, X_test, X_val
gc.collect()

#WINDOWING
print("windowing...")

window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=False)

X_train_ae, X_train_seq, y_train_w = window_builder.transform(X_train_proc, y_train_bin)
X_test_ae, X_test_seq, y_test_w = window_builder.transform(X_test_proc, y_test_bin)
X_val_ae, X_val_seq, y_val_w = window_builder.transform(X_val_proc, y_val_bin)
train_source_w = _window_last(train_source, WINDOW_SIZE, WINDOW_STEP)

# Free intermediate arrays — no longer needed
del X_train_proc, X_test_proc, X_val_proc, train_source
gc.collect()

print(f"    Shapes: train={X_train_seq.shape}   |   test={X_test_seq.shape}")

y_val_ohe = to_categorical(y_val_w, num_classes=2)
val_pack = (
    {"ae_input": X_val_ae, "cnn_input": X_val_seq, "lstm_input": X_val_seq},
    {"classification": y_val_ohe, "reconstruction": X_val_ae},
)

main_ckpt = MODELS_PATH / "best_model_binary.keras"
checkpoint = ModelCheckpoint(filepath=str(main_ckpt), monitor="val_loss", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

rng = np.random.default_rng(RANDOM_SEED)
hard_negative_history = []

print("\n=== initial training ===")
initial_focus_w, initial_focus_stats = build_benign_focus_weights(y_train_w, train_source_w)
print(
    f"normal_weight={NORMAL_WEIGHT}, attack_weight={ATTACK_WEIGHT}, "
    f"focal gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}, seed={RANDOM_SEED}, "
    f"window_size={WINDOW_SIZE}, window_step={WINDOW_STEP}, "
    f"benign_focus_source='{BENIGN_FOCUS_SOURCE or 'disabled'}', benign_focus_mult={BENIGN_FOCUS_MULT}"
)
if initial_focus_stats["enabled"]:
    print(
        f"  benign-focus: boosted {initial_focus_stats['boosted_count']} / "
        f"{initial_focus_stats['eligible_count']} benign windows"
    )
train_dataset = create_balanced_tf_dataset(
    X_train_ae,
    X_train_seq,
    to_categorical(y_train_w, num_classes=2),
    batch_size=BATCH_SIZE,
    attack_weight=ATTACK_WEIGHT,
    normal_weight=NORMAL_WEIGHT,
    shuffle_seed=RANDOM_SEED,
    extra_sample_weight=initial_focus_w,
)
model = build_model_compile(num_features)
steps_per_epoch = max(1, len(X_train_ae) // BATCH_SIZE)
model.fit(
    train_dataset,
    validation_data=val_pack,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint, reduce_lr],
    verbose=1,
)

# for hn_round in range(HARD_NEGATIVE_REFINEMENT_LOOPS):
#     print(f"\n=== hard-negative refinement {hn_round + 1}/{HARD_NEGATIVE_REFINEMENT_LOOPS} ===")
#     best_model = load_model(str(main_ckpt), compile=False)
#     iso_mine, _, val_scores_mine = fit_iso_and_scores(best_model, X_val_ae, X_val_seq, y_val_w)
#     t_mine, sel_mine = select_threshold(
#         y_val_w,
#         val_scores_mine,
#         mode=THRESHOLD_SELECTION_MODE,
#         min_normal_recall=MIN_NORMAL_RECALL_TARGET,
#         min_attack_recall_target=MIN_ATTACK_RECALL_TARGET,
#         min_attack_recall_floor=MIN_ATTACK_RECALL_FLOOR,
#         grid_start=THRESHOLD_GRID_START,
#         grid_end=THRESHOLD_GRID_END,
#         grid_step=THRESHOLD_GRID_STEP,
#     )
#     print(f"  mining threshold (provisional): {t_mine:.4f}  mode={sel_mine['mode']}")
#     X_ae_aug, X_seq_aug, y_aug, src_aug, mine_stats = mine_hard_negatives(
#         best_model, X_train_ae, X_train_seq, y_train_w, train_source_w, iso_mine, t_mine, rng
#     )
#     print(
#         f"  mined FP={mine_stats['fp_train']} (used {mine_stats['fp_used']}), "
#         f"FN={mine_stats['fn_train']} (used {mine_stats['fn_used']}); "
#         f"augmented train size={len(y_aug)}"
#     )
#     round_focus_w, round_focus_stats = build_benign_focus_weights(y_aug, src_aug)
#     if round_focus_stats["enabled"]:
#         print(
#             f"  benign-focus (round): boosted {round_focus_stats['boosted_count']} / "
#             f"{round_focus_stats['eligible_count']} benign windows"
#         )
#     hard_negative_history.append(
#         {
#             "round": hn_round + 1,
#             "mining_threshold": float(t_mine),
#             "mining_threshold_mode": sel_mine["mode"],
#             **mine_stats,
#             "augmented_train_size": int(len(y_aug)),
#             "benign_focus": round_focus_stats,
#         }
#     )
#     if mine_stats["fp_used"] == 0 and mine_stats["fn_used"] == 0:
#         print("  no hard negatives found; skipping refinement fit for this round")
#         continue
#     train_dataset = create_balanced_tf_dataset(
#         X_ae_aug,
#         X_seq_aug,
#         to_categorical(y_aug, num_classes=2),
#         batch_size=BATCH_SIZE,
#         attack_weight=ATTACK_WEIGHT,
#         normal_weight=NORMAL_WEIGHT,
#         shuffle_seed=RANDOM_SEED + hn_round + 1,
#         extra_sample_weight=round_focus_w,
#     )
#     model = load_model(str(main_ckpt), compile=False)
#     compile_model(model)
#     steps_ref = max(1, len(X_ae_aug) // BATCH_SIZE)
#     model.fit(
#         train_dataset,
#         validation_data=val_pack,
#         epochs=HARD_NEGATIVE_EPOCHS,
#         steps_per_epoch=steps_ref,
#         callbacks=[checkpoint, reduce_lr],
#         verbose=1,
#     )

print("\n=== final calibration & threshold ===")
best_model = load_model(str(main_ckpt), compile=False)
iso, _, y_val_scores = fit_iso_and_scores(best_model, X_val_ae, X_val_seq, y_val_w)
calibration_tag = "isotonic" if iso is not None else "none"
if iso is not None:
    joblib.dump(iso, MODELS_PATH / CALIBRATOR_FILENAME)
else:
    stale = MODELS_PATH / CALIBRATOR_FILENAME
    if stale.exists():
        stale.unlink()

best_threshold, sel_meta = select_threshold(
    y_val_w,
    y_val_scores,
    mode=THRESHOLD_SELECTION_MODE,
    min_normal_recall=MIN_NORMAL_RECALL_TARGET,
    min_attack_recall_target=MIN_ATTACK_RECALL_TARGET,
    min_attack_recall_floor=MIN_ATTACK_RECALL_FLOOR,
    grid_start=THRESHOLD_GRID_START,
    grid_end=THRESHOLD_GRID_END,
    grid_step=THRESHOLD_GRID_STEP,
)

print(
    f"     {'Thresh':>6} | {'Attack Rec':>10} | {'Normal Rec':>10} | "
    f"{'BalAcc':>8} | {'MacroF1':>8}"
)
print(f"     {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
for thresh in np.arange(0.05, 0.96, 0.05):
    y_pred = (y_val_scores > thresh).astype(int)
    report = classification_report(y_val_w, y_pred, output_dict=True, zero_division=0)
    attack_rec = report["1"]["recall"]
    normal_rec = report["0"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    bal = 0.5 * (normal_rec + attack_rec)
    mark = "*" if thresh <= best_threshold < thresh + 0.05 else " "
    print(
        f"  {mark}{thresh:>6.2f} | {attack_rec:>10.3f} | {normal_rec:>10.3f} | "
        f"{bal:>8.3f} | {macro_f1:>8.3f}"
    )

print(
    f"\nselected threshold: {best_threshold:.4f}  "
    f"(mode={sel_meta['mode']}, calibration={calibration_tag})"
)
print(
    f"  val @ threshold: Normal R={sel_meta['val_normal_recall']:.3f}, "
    f"Attack R={sel_meta['val_attack_recall']:.3f}"
)
if "val_macro_f1_ceiling" in sel_meta:
    c = sel_meta["val_macro_f1_ceiling"]
    print(
        f"  val macro-F1 ceiling (any t): {c['val_macro_f1_max']:.4f} @ t={c['threshold_at_val_macro_f1_max']:.3f} "
        f"(Normal R={c['val_normal_recall_at_macro_max']:.3f}, Attack R={c['val_attack_recall_at_macro_max']:.3f})"
    )
if sel_meta.get("dual_constraint_note"):
    print(f"  note: {sel_meta['dual_constraint_note']}")

training_config = {
    "seed": RANDOM_SEED,
    "window_size": WINDOW_SIZE,
    "window_step": WINDOW_STEP,
    "normal_weight": NORMAL_WEIGHT,
    "attack_weight": ATTACK_WEIGHT,
    "focal_gamma": FOCAL_GAMMA,
    "focal_alpha": FOCAL_ALPHA,
    "epochs_initial": EPOCHS,
    # "hard_negative_refinement_loops": HARD_NEGATIVE_REFINEMENT_LOOPS,
    # "hard_negative_epochs": HARD_NEGATIVE_EPOCHS,
    # "hard_negative_max_per_class": HARD_NEGATIVE_MAX_PER_CLASS,
    # "hard_negative_duplicates_fp": HARD_NEGATIVE_DUPLICATES_FP,
    # "hard_negative_duplicates_fn": HARD_NEGATIVE_DUPLICATES_FN,
    "benign_focus_source": BENIGN_FOCUS_SOURCE,
    "benign_focus_match": BENIGN_FOCUS_MATCH,
    "benign_focus_multiplier": BENIGN_FOCUS_MULT,
    "benign_focus_initial": initial_focus_stats,
    "hard_negative_rounds": hard_negative_history,
}

threshold_payload = {
    "threshold": float(best_threshold),
    "calibration": calibration_tag,
    "threshold_selection_mode": THRESHOLD_SELECTION_MODE,
    "min_normal_recall_target": MIN_NORMAL_RECALL_TARGET,
    "min_attack_recall_target": MIN_ATTACK_RECALL_TARGET,
    "min_attack_recall_floor": MIN_ATTACK_RECALL_FLOOR,
    "threshold_selection": sel_meta,
    "training_config": training_config,
    "target_macro_f1": TARGET_MACRO_F1,
}
with open(MODELS_PATH / "threshold.json", "w") as f:
    json.dump(threshold_payload, f, indent=2)

#EVALUATION

print("evaluating model...")
y_test_raw = best_model.predict({"ae_input": X_test_ae, "cnn_input": X_test_seq, "lstm_input": X_test_seq}, verbose=0)["classification"][:, 1]

if iso is not None:
    y_test_scores = iso.predict(y_test_raw)
else:
    y_test_scores = y_test_raw

y_test_pred = (y_test_scores > best_threshold).astype(int)

report = classification_report(
    y_true=y_test_w,
    y_pred=y_test_pred,
    target_names=["Normal", "Attack"],
    zero_division=0,
    output_dict=True,
) 

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH / "classification_report.csv")

cm_test = confusion_matrix(y_test_w, y_test_pred)
pd.DataFrame(cm_test, index=["true_Normal", "true_Attack"], columns=["pred_Normal", "pred_Attack"]).to_csv(
    REPORTS_PATH / "confusion_matrix_test.csv"
)

eval_summary = {
    "calibration": calibration_tag,
    "threshold": float(best_threshold),
    "threshold_selection_mode": THRESHOLD_SELECTION_MODE,
    "threshold_selection": sel_meta,
    "training_config": training_config,
    "test": {
        "accuracy": float(report["accuracy"]),
        "normal_precision": float(report["Normal"]["precision"]),
        "normal_recall": float(report["Normal"]["recall"]),
        "normal_f1": float(report["Normal"]["f1-score"]),
        "attack_precision": float(report["Attack"]["precision"]),
        "attack_recall": float(report["Attack"]["recall"]),
        "attack_f1": float(report["Attack"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    },
    "test_confusion_matrix": cm_test.tolist(),
}
with open(REPORTS_PATH / "eval_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print(f"\nCLASSIFICATION REPORT:")
print(report_df.to_string())

print(f"\n{'='*60}")
print(f"\n  Stage 1 Results")
print(f"\n  Threshold: {best_threshold:.2f}")
print(f"  Normal Recall: {report['Normal']['recall']:.4f}")
print(f"  Attack Recall:  {report['Attack']['recall']:.4f}")
print(f"  Macro F1:       {report['macro avg']['f1-score']:.4f}")
print(f"\n{'='*60}")
