#!/usr/bin/env python3
"""
Standalone feature selection analysis script.
Run: python -m src.preprocessing.analyze_features
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.boruta_shap_selector import (
    FeaturePreFilter,
    BorutaSHAPSelector,
    ShapRFESelector,
    HybridFeatureSelector,
)
from src.preprocessing.encoding import CategoricalEncoder


def analyze_leaky_features(X: pd.DataFrame, y: pd.Series, dataset: str):
    print("\n" + "=" * 60)
    print("ANALYZING POTENTIALLY LEAKY/REDUNDANT FEATURES")
    print("=" * 60)

    prefilter = FeaturePreFilter(dataset=dataset)
    report = prefilter.fit(X).get_report()

    print(f"\nFeatures to be removed ({len(report['removed'])}):")
    for feat in report["removed"]:
        print(f"  - {feat}")

    print(f"\nFeatures to keep ({len(report['kept'])}):")
    for feat in report["kept"]:
        print(f"  + {feat}")


def analyze_correlations(X: pd.DataFrame, threshold: float = 0.95):
    print("\n" + "=" * 60)
    print("HIGHLY CORRELATED FEATURE PAIRS")
    print("=" * 60)

    numeric_df = X.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [
        (col, idx, val)
        for col in upper.columns
        for idx, val in upper[col].items()
        if val > threshold
    ]

    if high_corr:
        print(f"\nPairs with correlation > {threshold}:")
        for feat1, feat2, corr in sorted(high_corr, key=lambda x: -x[2]):
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print(f"\nNo pairs with correlation > {threshold}")


def analyze_variance(X: pd.DataFrame, threshold: float = 0.01):
    print("\n" + "=" * 60)
    print("LOW VARIANCE FEATURES")
    print("=" * 60)

    numeric_df = X.select_dtypes(include=[np.number])
    variances = numeric_df.var()
    low_var = variances[variances < threshold]

    if len(low_var) > 0:
        print(f"\nFeatures with variance < {threshold}:")
        for feat, var in low_var.items():
            print(f"  - {feat}: {var:.6f}")
    else:
        print(f"\nNo features with variance < {threshold}")


def run_boruta_analysis(
    X: pd.DataFrame, y, dataset: str, n_trials: int = 50, sample: int = 50000
):
    print("\n" + "=" * 60)
    print("BORUTA-SHAP FEATURE SELECTION")
    print("=" * 60)

    if len(X) > sample:
        print(f"Subsampling to {sample:,} samples for Boruta...")
        idx = np.random.choice(len(X), sample, replace=False)
        X_sub = X.iloc[idx].copy()
        y_sub = y.iloc[idx].copy() if isinstance(y, pd.Series) else y[idx]
    else:
        X_sub = X.copy()
        y_sub = y

    encoder = CategoricalEncoder()
    encoder.fit(X_sub)
    X_sub = encoder.transform(X_sub)

    label_encoder = LabelEncoder()
    y_sub = label_encoder.fit_transform(y_sub)

    X_sub = X_sub.replace([np.inf, -np.inf], np.nan)
    X_sub = X_sub.fillna(0)

    selector = BorutaSHAPSelector(
        n_trials=n_trials,
        sample=False,
        model_type="rf",
        verbose=True,
    )

    selector.fit(X_sub, y_sub)

    print("\n--- Selection Results ---")
    print(f"Accepted ({len(selector.accepted_)}): {selector.accepted_}")
    print(f"Tentative ({len(selector.tentative_)}): {selector.tentative_}")
    print(f"Rejected ({len(selector.rejected_)}): {selector.rejected_}")

    return selector


def run_hybrid_analysis(X: pd.DataFrame, y, dataset: str, n_trials: int = 50):
    print("\n" + "=" * 60)
    print("HYBRID FEATURE SELECTION (Pre-filter + Boruta + RFE)")
    print("=" * 60)

    X_copy = X.copy()
    encoder = CategoricalEncoder()
    encoder.fit(X_copy)
    X_copy = encoder.transform(X_copy)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
    X_copy = X_copy.fillna(0)

    selector = HybridFeatureSelector(
        dataset=dataset,
        n_trials=n_trials,
        use_rfe_tuning=True,
        verbose=True,
    )

    selector.fit(X_copy, y_enc)

    print("\n--- Final Selection ---")
    print(f"Selected features ({len(selector.selected_features_)}):")
    for i, feat in enumerate(selector.selected_features_, 1):
        print(f"  {i}. {feat}")

    return selector


def main():
    parser = argparse.ArgumentParser(description="Analyze and select features")
    parser.add_argument("--dataset", type=str, default="UNSW", choices=["UNSW", "CIC"])
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["analyze", "boruta", "hybrid", "all"],
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of Boruta trials"
    )
    parser.add_argument(
        "--sample", type=int, default=50000, help="Max samples for analysis"
    )

    args = parser.parse_args()

    if args.dataset == "UNSW":
        data_path = Path("data/processed/UNSW-NB15/splits/train.csv")
    else:
        data_path = Path("data/processed/CIC-IDS2017/splits/train/data.parquet")

    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print("Run preprocessing first to generate data")
        return

    print(f"Loading {data_path}...")
    if args.dataset == "UNSW":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)

    print(f"Dataset shape: {df.shape}")

    if args.dataset == "UNSW":
        label_col = "attack_cat"
        drop_cols = ["attack_cat", "label", "id"]
    else:
        label_col = "attack_type"
        drop_cols = ["Label", "attack_label", "attack_type", "source_file"]

    y = df[label_col]
    X = df.drop(columns=drop_cols, errors="ignore")

    if args.mode in ["analyze", "all"]:
        analyze_leaky_features(X, y, args.dataset)
        analyze_correlations(X)
        analyze_variance(X)

    if args.mode in ["boruta", "all"]:
        run_boruta_analysis(
            X, y, args.dataset, n_trials=args.n_trials, sample=args.sample
        )

    if args.mode in ["hybrid", "all"]:
        run_hybrid_analysis(X, y, args.dataset, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
