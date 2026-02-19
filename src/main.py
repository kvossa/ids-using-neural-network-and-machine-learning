import pandas as pd
from preprocessing.split import split_and_store
from preprocessing.pipeline import IDSPipeline
from preprocessing.scaling import MultiClassLabelEncoder
from sklearn.model_selection import train_test_split

def load_unsw(path: str) -> df:
    return pd.read_csv(path)

def load_cic(path:str) -> df:
    return pd.read_parquet(path)

def split(data:df):
    return

def call(data:df, dataset:str):
    X = data.drop("label", axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    label_encoder = MultiClassLabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    preprocessing_pipeline = IDSPipeline()
    preprocessing_pipeline.build_pipeline()

    X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train_enc)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    




    