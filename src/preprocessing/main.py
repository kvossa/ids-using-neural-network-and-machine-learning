import joblib
import pandas as pd
from pipeline import IDSPipeline
from scaling import MultiClassLabelEncoder
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def load_unsw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_cic(path:str) -> pd.DataFrame:
    return pd.read_parquet(path)

def call(train_path:Path, test_path:Path, val_path:Path, dataset:str, stratify_column:str) -> set:
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

    y_train = X_train[stratify_column]
    X_train = X_train.drop(stratify_column, axis=1)

    y_test = X_test[stratify_column]
    X_test = X_test.drop(stratify_column, axis=1)

    y_val = X_val[stratify_column]
    X_val = X_val.drop(stratify_column, axis=1)

    # label_encoder = MultiClassLabelEncoder( target_col=stratify_column)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    y_val_enc = label_encoder.transform(y_val)


    preprocessing_pipeline = IDSPipeline(dataset=dataset)
    preprocessing_pipeline.build_pipeline()

    X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train_enc)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    X_val_processed = preprocessing_pipeline.transform(X_val)

    print("##shape##")
    print(X_train_processed.shape)
    print(X_test_processed.shape)

    print("##summary##")
    print(X_train_processed.isna().sum().sum())
    print(X_train_processed.dtypes.unique())

    print("##are the same? trainset vs testset##")
    print(X_train_processed.shape[1] == X_test_processed.shape[1])

    print("##Y data##")
    print(set(y_train_enc))
    print(set(y_test_enc))

    print("##balance##")
    print(pd.Series(y_train_enc).value_counts(normalize=True))

    model_path = f'../../models/preprocessing/{dataset.lower}/preprocessing.pkl'
    joblib.dump(preprocessing_pipeline, model_path)

    print(f"preprocessing model saved in {model_path}")

    return (X_train_processed, y_train_enc, X_test_processed, y_test_enc, X_val_processed, y_val_enc)


if __name__ == "__main__":
    #CIC
    dataset: str = "CIC"
    train_path: Path = Path('../../data/processed/CIC-IDS2017/splits/train/data.parquet')
    test_path: Path = Path('../../data/processed/CIC-IDS2017/splits/test/data.parquet')
    val_path: Path = Path('../../data/processed/CIC-IDS2017/splits/val/data.parquet')
    stratify_column:str = 'attack_type'
    #UNSW
    # dataset: str = "UNSW"
    # train_path: Path = Path('../../data/processed/UNSW-NB15/splits/train.csv')
    # test_path: Path = Path('../../data/processed/UNSW-NB15/splits/test.csv')
    # val_path: Path = Path('../../data/processed/UNSW-NB15/splits/validation.csv')
    # stratify_column = 'attack_cat'


    call(train_path=train_path, test_path=test_path, val_path=val_path, dataset='CIC', stratify_column=stratify_column)



    