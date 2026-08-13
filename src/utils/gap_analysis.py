import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATHS, PREPROC_PATHS


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


DATASET_CONFIGS = {
    "CIC": {
        "preprocessor_path": PREPROC_PATHS["cic"]["binary_preprocessor"],
        "train_path": DATA_PATHS["cic"]["train"],
        "feature_names_path": PREPROC_PATHS["cic"]["feature_names"],
        "window_size": 5,
        "train_label_col": "attack_type",
        "train_benign_label": "BENIGN",
        "train_drop_cols": ["Label", "attack_label", "attack_type", "source_file"],
        "n_features": 41,
    },
    "UNSW": {
        "preprocessor_path": PREPROC_PATHS["unsw"]["binary_preprocessor"],
        "train_path": DATA_PATHS["unsw"]["train"],
        "feature_names_path": PREPROC_PATHS["unsw"]["feature_names"],
        "window_size": 10,
        "train_label_col": "attack_cat",
        "train_benign_label": "Normal",
        "train_drop_cols": ["attack_cat", "label", "id"],
        "n_features": 193,
    },
}


def load_training_features(cfg: dict, sample: int = 20000) -> dict:
    ppath = cfg["preprocessor_path"]
    fpath = cfg["train_path"]
    drop_cols = cfg["train_drop_cols"]
    label_col = cfg["train_label_col"]
    benign_label = cfg["train_benign_label"]

    if not Path(ppath).is_file():
        raise FileNotFoundError(f"Preprocessor not found: {ppath}")
    if not Path(fpath).is_file():
        raise FileNotFoundError(f"Training data not found: {fpath}")

    import joblib
    pre = joblib.load(ppath)

    if fpath.endswith(".parquet"):
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, low_memory=False)

    labels = df[label_col].values
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    X_proc = pre.transform(X)
    X_proc = np.asarray(X_proc, dtype=np.float32)

    if len(X_proc) > sample:
        idx = np.random.RandomState(42).choice(len(X_proc), sample, replace=False)
        X_proc = X_proc[idx]
        labels = labels[idx]

    benign_mask = labels == benign_label
    attack_mask = ~benign_mask

    return {
        "full": X_proc,
        "benign": X_proc[benign_mask],
        "attack": X_proc[attack_mask],
        "labels": labels,
        "n_total": len(X_proc),
        "n_benign": benign_mask.sum(),
        "n_attack": attack_mask.sum(),
    }


def load_real_features(cfg: dict, csv_path: str) -> np.ndarray:
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"Real features CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    if len(feat_cols) != cfg["n_features"]:
        print(f"  WARNING: Expected {cfg['n_features']} features, found {len(feat_cols)} in {csv_path}")
    X = df[feat_cols].values.astype(np.float32)
    return X


def analyze_shift(train: dict, real: np.ndarray, feature_names: list, top_k: int = 20):
    X_train = train["full"]
    X_benign = train["benign"]
    X_attack = train["attack"]

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-10
    real_mean = real.mean(axis=0)
    real_std = real.std(axis=0) + 1e-10

    # Z-score shift: |mean_diff| / train_std
    mean_diff = real_mean - train_mean
    z_shift = np.abs(mean_diff) / train_std

    # KL-like divergence: distribution overlap score
    # Use (mean_diff)^2 / (train_std * real_std) — rough measure
    divergence = mean_diff ** 2 / (train_std * real_std)

    ranked = np.argsort(z_shift)[::-1]

    print(f"\n{'='*80}")
    print(f"TOP {top_k} MOST SHIFTED FEATURES (real vs training)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Feature':<35} {'Z-shift':<9} {'Train μ':<10} {'Real μ':<10} {'Train σ':<10} {'Real σ':<10}")
    print("-" * 80)

    for rank, idx in enumerate(ranked[:top_k], 1):
        fname = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        print(f"{rank:<5} {fname:<35} {z_shift[idx]:<9.2f} {train_mean[idx]:<10.6f} {real_mean[idx]:<10.6f} {train_std[idx]:<10.6f} {real_std[idx]:<10.6f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    features_above = (z_shift > 2.0).sum()
    features_above_3 = (z_shift > 3.0).sum()
    features_above_5 = (z_shift > 5.0).sum()
    print(f"  Total features: {X_train.shape[1]}")
    print(f"  Shift > 2σ: {features_above} ({features_above/X_train.shape[1]*100:.1f}%)")
    print(f"  Shift > 3σ: {features_above_3} ({features_above_3/X_train.shape[1]*100:.1f}%)")
    print(f"  Shift > 5σ: {features_above_5} ({features_above_5/X_train.shape[1]*100:.1f}%)")

    # Top affected features by type
    print(f"\n{'='*80}")
    print("TOP 5 FEATURES WITH LARGEST ABSOLUTE DIFFERENCE")
    print(f"{'='*80}")
    abs_ranked = np.argsort(np.abs(mean_diff))[::-1]
    for rank, idx in enumerate(abs_ranked[:5], 1):
        fname = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        print(f"  {rank}. {fname:<35}  Δμ = {mean_diff[idx]:+.6f}  (train={train_mean[idx]:.4f} → real={real_mean[idx]:.4f})")

    return z_shift, mean_diff


def pca_projection(train: dict, real: np.ndarray, feature_names: list, dataset: str):
    try:
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[SKIP] PCA plot requires sklearn + matplotlib")
        return

    X_train = train["full"]
    labels = train["labels"]

    n_samples_real = len(real)
    n_samples_train = min(5000, len(X_train))
    idx = np.random.RandomState(42).choice(len(X_train), n_samples_train, replace=False)
    X_sample = X_train[idx]
    label_sample = labels[idx]

    X_all = np.vstack([X_sample, real[:min(5000, n_samples_real)]])
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(12, 8))

    train_2d = X_2d[:n_samples_train]
    real_2d = X_2d[n_samples_train:n_samples_train + min(5000, n_samples_real)]

    ax.scatter(train_2d[:, 0], train_2d[:, 1], c="blue", alpha=0.3, s=5, label="Training", edgecolors="none")
    ax.scatter(real_2d[:, 0], real_2d[:, 1], c="red", alpha=0.6, s=20, label="Real (live)", marker="X", edgecolors="black")

    ax.set_title(f"{dataset} — PCA: Training vs Real Traffic")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend()
    ax.grid(alpha=0.3)

    outpath = f"reports/figures/gap_analysis_{dataset.lower()}.png"
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPCA plot saved: {outpath}")
    plt.close(fig)

    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Gap analysis: training vs real-time feature distributions")
    parser.add_argument("--dataset", choices=["CIC", "UNSW"], required=True)
    parser.add_argument("--features-csv", type=str, required=True, help="Path to captured features CSV")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top features to show")
    parser.add_argument("--pca", action="store_true", help="Generate PCA 2D projection")
    parser.add_argument("--train-sample", type=int, default=20000, help="Training samples to use")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]

    print(f"Loading training features for {args.dataset}...")
    train = load_training_features(cfg, sample=args.train_sample)
    print(f"  Training: {train['n_total']} total ({train['n_benign']} benign, {train['n_attack']} attack)")

    print(f"Loading real features from {args.features_csv}...")
    real = load_real_features(cfg, args.features_csv)
    print(f"  Real: {len(real)} samples, {real.shape[1]} features")

    feature_names = np.load(cfg["feature_names_path"], allow_pickle=True).tolist()
    print(f"  Feature names loaded: {len(feature_names)}")

    if args.dataset == "CIC":
        print("\nNOTE: CIC binary preproc produces 41 features. Stage 2 uses multiclass (78).")
        print("      These 41 are the AE+CNN+LSTM inputs. Shifts here affect both stages.")

    analyze_shift(train, real, feature_names, top_k=args.top_k)

    if args.pca:
        pca_projection(train, real, feature_names, args.dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
