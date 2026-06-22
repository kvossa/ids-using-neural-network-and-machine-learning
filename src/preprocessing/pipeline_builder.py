import pandas as pd
from sklearn.pipeline import Pipeline
from src.preprocessing.pipeline.features_extraction import FeatureExtraction
from src.preprocessing.pipeline.clean import DataCleaner
from src.preprocessing.pipeline.scaling import DataScaler, MultiClassLabelEncoder
from src.preprocessing.pipeline.encoding import CategoricalEncoder
from src.preprocessing.pipeline.imputer import DataFrameImputer
from src.preprocessing.pipeline.column_dropper import ColumnDropper
from src.preprocessing.pipeline.log_transformer import LogTransformer


class IDSPipeline:
    def __init__(self, dataset="CIC"):
        self.dataset = dataset
        self.pipeline = None

    def build_pipeline(self):
        steps = [
            ("cleaner", DataCleaner()),
            ("feature_extraction", FeatureExtraction(dataset=self.dataset)),
            ("column_dropper", ColumnDropper(dataset=self.dataset)),
            ("categorical_encoder", CategoricalEncoder(handle_unknown="ignore")),
            ("imputer", DataFrameImputer(strategy="median")),
            ("log_transformer", LogTransformer()),
            ("robust_scaler", DataScaler(scaler_type="robust")),
            ("minmax_scaler", DataScaler(scaler_type="minmax")),
        ]

        self.pipeline = Pipeline(steps)

        return self.pipeline

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def transform(self, X):
        result = self.pipeline.transform(X)
        if hasattr(result, 'values'):
            result = result.values
        return result.astype('float32') if result.dtype != 'float32' else result

    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test):
        return self.pipeline.predict_proba(X_test)


# if __name__ == "__main__":
# 	cic_df = pd.read_parquet('../../data/cic_ids2017.parquet')
# 	unsw_df = pd.read_csv('data/unsw_nb15.csv')
