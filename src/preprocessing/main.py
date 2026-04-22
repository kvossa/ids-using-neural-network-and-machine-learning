import joblib
import pandas as pd
from src.preprocessing.pipeline import IDSPipeline
from src.preprocessing.scaling import MultiClassLabelEncoder
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

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

    label_encoder_path = (
        f"models/preprocessing/binary/{dataset.lower()}/label_encoder.pkl"
    )
    joblib.dump(label_encoder, label_encoder_path)

    print(f"label encoder saved in {label_encoder_path}")

    # multiclass/binary
    model_path = f"models/preprocessing/binary/{dataset.lower()}/preprocessing.pkl"
    joblib.dump(preprocessing_pipeline, model_path)

    print(f"preprocessing model saved in {model_path}")

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
        choices=["hybrid", "boruta", "rfe", "original"],
        help="Feature selector type",
    )
    args = parser.parse_args()

    selector_type = args.selector if args.selector != "original" else "original"

    # UNSW
    dataset: str = "UNSW"
    train_path: Path = Path("data/processed/UNSW-NB15/splits/train.csv")
    test_path: Path = Path("data/processed/UNSW-NB15/splits/test.csv")
    val_path: Path = Path("data/processed/UNSW-NB15/splits/validation.csv")
    stratify_column = "attack_cat"

    call(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        dataset=dataset,
        stratify_column=stratify_column,
        selector_type=selector_type,
    )
