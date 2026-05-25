import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self.feature_names_ = None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_ = X.columns.tolist()
        self.imputer.fit(X)
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            self.feature_names_ = X.columns.tolist()
        
        X_imputed = self.imputer.transform(X)
        if self.feature_names_ is not None:
            return pd.DataFrame(X_imputed, columns=self.feature_names_, index=X.index if hasattr(X, "index") else None)
        return X_imputed
