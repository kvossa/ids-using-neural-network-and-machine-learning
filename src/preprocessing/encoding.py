import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns=None, handle_unknown='error'):
        self.categorical_columns = categorical_columns
        self.handle_unknown = handle_unknown
        # self.encoders_ = {}
        self.fitted_ = False

    def fit(self, X, y=None):
        X_copy = X.copy()

        # if isinstance(X_copy, pd.DataFrame):
            # y = X_copy.squeeze()

        if self.categorical_columns is None:
            self.categorical_columns = X_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        
        X_copy[self.categorical_columns] = X_copy[self.categorical_columns].fillna("Missing").astype(str)
        
        self.encoder_ = OneHotEncoder(handle_unknown = self.handle_unknown, sparse_output=False)

        if len(self.categorical_columns) > 0:
            self.encoder_.fit(X_copy[self.categorical_columns])
            self.features_out_ = None

        self.original_columns_ = X_copy.columns.tolist()
        self.fitted_ = True
        return self
    
    def transform(self, X):
        if not self.fitted_:
            raise ValueError("Encoder must be fitted before transform")
        
        # if isinstance(y, pd.DataFrame):
#         #     y = y.squeeze()

        X_encoded = X.copy()
        X_encoded = X_encoded.reindex(columns=self.original_columns_, fill_value="Missing")

        if len(self.categorical_columns) > 0:
            X_encoded[self.categorical_columns] = X_encoded[self.categorical_columns].fillna("Missing").astype(str)

            encoded_array = self.encoder_.transform(
                X_encoded[self.categorical_columns]
            )

            encoded_df = pd.DataFrame(
                encoded_array,
                columns=self.encoder_.get_feature_names_out(
                    self.categorical_columns
                ),
                index=X_encoded.index
            )

            X_encoded = X_encoded.drop(
                columns=self.categorical_columns
            )

            X_encoded = pd.concat(
                [X_encoded, encoded_df],
                axis=1
            )
        
        if self.features_out_ is None:
            self.features_out_ = X_encoded.columns.tolist()

        X_encoded = X_encoded.reindex(columns=self.features_out_, fill_value=0)

        return X_encoded
        
