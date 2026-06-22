import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.numeric_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        return self

    def transform(self, X):
        X_t = X.copy()
        for col in self.numeric_cols_:
            if col in X_t.columns:
                X_t[col] = np.log1p(X_t[col].clip(lower=0))
        return X_t
