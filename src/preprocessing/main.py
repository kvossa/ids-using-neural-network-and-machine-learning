import joblib
import pandas as pd
from src.preprocessing.pipeline.pipeline import IDSPipeline
from src.preprocessing.pipeline.scaling import MultiClassLabelEncoder
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import json
from typing import List, Optional

BINARY = True
NUMBER_FEATURES = 40

def load_unsw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_cic(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def call(
    train_path: Path,
    test_path: Path,
    val_path: Path,
    dataset: str,
    stratify_column: str,
    selector_type: str = "hybrid",
    manual_features: Optional[List[str]] = None,
) -> set:
    # X = data.drop(stratify_column, axis=1)
    # y = data[stratify_column]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    if dataset == "CIC":
        X_train = load_cic(train_path)
        X_test = load_cic(test_path)
        X_val = load_cic(val_path)
        drop_columns = ["Label", "attack_label", "attack_type", "source_file"]
        use_feature_selection = True
    elif dataset == "UNSW":
        X_train = load_unsw(train_path)
        X_test = load_unsw(test_path)
        X_val = load_unsw(val_path)
        drop_columns = ["attack_cat", "label", "id"]
        use_feature_selection = True

    y_train = X_train[stratify_column]
    X_train = X_train.drop(columns=drop_columns, axis=1)

    y_test = X_test[stratify_column]
    X_test = X_test.drop(columns=drop_columns, axis=1)

    y_val = X_val[stratify_column]
    X_val = X_val.drop(columns=drop_columns, axis=1)

    # label_encoder = MultiClassLabelEncoder( target_col=stratify_column)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    y_val_enc = label_encoder.transform(y_val)

    if BINARY:
        normal = "BENIGN" if dataset == "CIC" else "Normal"
        y_train_for_selection = (y_train != normal).astype(int).values
    else:
        y_train_for_selection = y_train_enc

    # print(y_train_for_selection.head())

    preprocessing_pipeline = IDSPipeline(
        dataset=dataset,
        use_feature_selection=use_feature_selection,
        k_features=NUMBER_FEATURES,
        selector_type=selector_type,
        manual_features=manual_features,
    )
    preprocessing_pipeline.build_pipeline()

    X_train_processed = preprocessing_pipeline.fit_transform(
        X_train, y_train_for_selection
    )
    X_test_processed = preprocessing_pipeline.transform(X_test)
    X_val_processed = preprocessing_pipeline.transform(X_val)

    print("##shape##")
    print(X_train_processed.shape)
    print(X_test_processed.shape)

    print("##types##")
    print(X_train_processed.isna().sum().sum())
    print(X_train_processed.dtypes.unique())

    print("##are the same? trainset vs testset##")
    print(X_train_processed.shape[1] == X_test_processed.shape[1])

    print("##Y data##")
    print(set(y_train_enc))
    print(set(y_test_enc))

    print("##balance##")
    print(pd.Series(y_train_for_selection).value_counts(normalize=True))

    label_encoder_multi_path = (
        f"models/preprocessing/multiclass/{dataset.lower()}/label_encoder.pkl"
    )
    joblib.dump(label_encoder, label_encoder_multi_path)
    print(f"label encoder (multiclass) saved in {label_encoder_multi_path}")

    binary_encoder_path = f"models/preprocessing/binary/{dataset.lower()}/label_encoder.pkl"
    Path(binary_encoder_path).parent.mkdir(parents=True, exist_ok=True)
    binary_label_encoder = LabelEncoder()
    normal = "BENIGN" if dataset == "CIC" else "Normal"
    y_binary = (y_train != normal).astype(int)
    binary_label_encoder.fit(y_binary)
    joblib.dump(binary_label_encoder, binary_encoder_path)
    print(f"label encoder (binary) saved in {binary_encoder_path}")

    model_multi_path = f"models/preprocessing/binary/{dataset.lower()}/preprocessing.pkl"
    joblib.dump(preprocessing_pipeline, model_multi_path)
    print(f"preprocessing (multiclass) saved in {model_multi_path}")

    if selector_type in {"fixed", "manual_fixed", "manual_hybrid"}:
        pipeline_binary_path = f"models/preprocessing/binary/{dataset.lower()}/preprocessing.pkl"
        joblib.dump(preprocessing_pipeline, pipeline_binary_path)
        print(f"preprocessing (binary) saved in {pipeline_binary_path}")

    return (
        X_train_processed,
        y_train_enc,
        X_test_processed,
        y_test_enc,
        X_val_processed,
        y_val_enc,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess IDS data")
    parser.add_argument(
        "--selector",
        type=str,
        default="hybrid",
        choices=["fixed", "hybrid", "boruta", "rfe", "manual_fixed", "manual_hybrid", "original"],
        help="Feature selector type",
    )
    parser.add_argument(
        "--manual-features",
        type=str,
        default="",
        help="Comma-separated list of manual features to keep (used by manual_* selectors).",
    )
    parser.add_argument(
        "--manual-features-file",
        type=str,
        default="",
        help="Path to JSON/TXT file with manual features (used by manual_* selectors).",
    )
    args = parser.parse_args()

    selector_type = args.selector if args.selector != "original" else "original"

    manual_features = None
    if args.manual_features:
        manual_features = [f.strip() for f in args.manual_features.split(",") if f.strip()]
    elif args.manual_features_file:
        p = Path(args.manual_features_file)
        if p.suffix.lower() == ".json":
            manual_features = json.loads(p.read_text())
        else:
            manual_features = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    # UNSW
    # dataset: str = "UNSW"
    # train_path: Path = Path("data/processed/UNSW-NB15/splits/train.csv")
    # test_path: Path = Path("data/processed/UNSW-NB15/splits/test.csv")
    # val_path: Path = Path("data/processed/UNSW-NB15/splits/validation.csv")
    # stratify_column = "attack_cat"

    # CIC
    dataset: str = "CIC"
    train_path: Path = Path("data/processed/CIC-IDS2017/splits/train/data.parquet")
    test_path: Path = Path("data/processed/CIC-IDS2017/splits/test/data.parquet")
    val_path: Path = Path("data/processed/CIC-IDS2017/splits/val/data.parquet")
    stratify_column = "attack_type"


    call(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        dataset=dataset,
        stratify_column=stratify_column,
        selector_type=selector_type,
        manual_features=manual_features,
    )
