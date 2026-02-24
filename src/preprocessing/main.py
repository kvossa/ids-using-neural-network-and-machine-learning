import joblib
import pandas as pd
from split import split_and_store
from pipeline import IDSPipeline
from scaling import MultiClassLabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_unsw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_cic(path:str) -> pd.DataFrame:
    return pd.read_parquet(path)

def call(train_path:Path, test_path:Path, val_path:Path, dataset:str, stratify_column:str):
    # X = data.drop(stratify_column, axis=1)
    # y = data[stratify_column]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    if dataset == 'CIC':
        X_train = load_cic(train_path)
        X_test = load_cic(test_path)
        X_val = load_cic(val_path)
    elif dataset == 'UNSW':
        X_train = load_unsw(train_path)
        X_test = load_unsw(test_path)
        X_val = load_unsw(val_path)

    label_encoder = MultiClassLabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    preprocessing_pipeline = IDSPipeline(dataset=dataset)
    preprocessing_pipeline.build_pipeline()

    X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train_enc)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    joblib.dump(preprocessing_pipeline, '../../models')

def call_for_unsw(train_df, test_df):
    train_


if __name__ == "__main__":
    #CIC
    dataset: str = "CIC"
    train_path: Path = Path('../../data/processed/CIC-IDS2017/splits/train/data.parquet')
    test_path: Path = Path('../../data/processed/CIC-IDS2017/splits/test/data.parquet')
    val_path: Path = Path('../../data/processed/CIC-IDS2017/splits/val/data.parquet')
    stratify_column = 'attack_type'
    #UNSW
    dataset: str = "UNSW"
    train_path: Path = Path('../../data/processed/UNSW-NB15/splits/train.csv')
    test_path: Path = Path('../../data/processed/UNSW-NB15/splits/test.csv')
    val_path: Path = Path('../../data/processed/UNSW-NB15/splits/validation.csv')
    stratify_column = 'attack_cat'




    