import joblib
import pandas as pd
from src.preprocessing.pipeline_builder import IDSPipeline
from src.preprocessing.pipeline.scaling import MultiClassLabelEncoder
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import json
from typing import List, Optional
from src.config import DATA_PATHS, PREPROC_PATHS


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
) -> set:
    if dataset == "CIC":
        X_train = load_cic(train_path)
        X_test = load_cic(test_path)
        X_val = load_cic(val_path)
        drop_columns = ["Label", "attack_label", "attack_type", "source_file"]
    elif dataset == "UNSW":
        X_train = load_unsw(train_path)
        X_test = load_unsw(test_path)
        X_val = load_unsw(val_path)
        drop_columns = ["attack_cat", "label", "id"]

    y_train = X_train[stratify_column]
    X_train = X_train.drop(columns=drop_columns, axis=1)

    y_test = X_test[stratify_column]
    X_test = X_test.drop(columns=drop_columns, axis=1)

    y_val = X_val[stratify_column]
    X_val = X_val.drop(columns=drop_columns, axis=1)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    y_val_enc = label_encoder.transform(y_val)

    normal = "BENIGN" if dataset == "CIC" else "Normal"
    y_train_for_selection = (y_train != normal).astype(int).values

    preprocessing_pipeline = IDSPipeline(dataset=dataset)
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

    label_encoder_multi_path = PREPROC_PATHS[dataset.lower()]["multiclass_encoder"]
    joblib.dump(label_encoder, label_encoder_multi_path)
    print(f"label encoder (multiclass) saved in {label_encoder_multi_path}")

    binary_encoder_path = PREPROC_PATHS[dataset.lower()]["binary_encoder"]
    Path(binary_encoder_path).parent.mkdir(parents=True, exist_ok=True)
    binary_label_encoder = LabelEncoder()
    y_binary = (y_train != normal).astype(int)
    binary_label_encoder.fit(y_binary)
    joblib.dump(binary_label_encoder, binary_encoder_path)
    print(f"label encoder (binary) saved in {binary_encoder_path}")

    model_path = PREPROC_PATHS[dataset.lower()]["binary_preprocessor"]
    joblib.dump(preprocessing_pipeline, model_path)
    print(f"preprocessing saved in {model_path}")

    return (
        X_train_processed,
        y_train_enc,
        X_test_processed,
        y_test_enc,
        X_val_processed,
        y_val_enc,
    )


if __name__ == "__main__":
    dataset: str = "CIC"
    train_path: Path = Path(DATA_PATHS["cic"]["train"])
    test_path: Path = Path(DATA_PATHS["cic"]["test"])
    val_path: Path = Path(DATA_PATHS["cic"]["val"])
    stratify_column = "attack_type"

    call(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        dataset=dataset,
        stratify_column=stratify_column,
    )
