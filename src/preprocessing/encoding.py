
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns=None, handle_unknown='error'):
        self.categorical_columns = categorical_columns
        self.handle_unknown = handle_unknown
        # self.encoders_ = {}
        # self.fitted_ = False
        
    def fit(self, X, y=None):
        if self.categorical_columns is None:
            self.categorical_columns = X.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        
        self.encoder_ = OneHotEncoder(
            handle_unknown = self.handle_unknown,
            sparse_output=False
        )

        if len(self.categorical_columns) > 0:
            self.encoder_.fit(X[self.categorical_columns])

        return self
        # for col in self.categorical_columns:
        #     if col in X.columns:
        #         le = LabelEncoder()
        #         # Handle NaN values
        #         X_clean = X[col].fillna('MISSING')
        #         le.fit(X_clean.astype(str))
        #         self.encoders_[col] = le
        #         print(f"  - {col}: {len(le.classes_)} categories")
                
        # self.fitted_ = True
    
    def transform(self, X):
        X_encoded = X.copy()

        if len(self.categorical_columns) > 0:
            encoded_array = self.encoder_.transform(
                X_encoded[self.categorical_columns]
            )

            encoded_df = pd.DataFrame(
                encoded_array,
                columns=self.encoder_.get_feature_names_out(
                    self.categorical_columns
                ),
                index=X.index
            )

            X_encoded = X_encoded.drop(
                columns=self.categorical_columns
            )

            X_encoded = pd.concat(
                [X_encoded, encoded_df],
                axis=1
            )

        return X_encoded
        
        # if not self.fitted_:
        #     raise ValueError("Encoder must be fitted before transform")
            
        # for col, encoder in self.encoders_.items():
        #     if col in X_encoded.columns:
        #         # Handle unknown categories
        #         X_clean = X_encoded[col].fillna('MISSING').astype(str)
                
        #         if self.handle_unknown == 'error':
        #             # Check for unknown categories
        #             known_classes = set(encoder.classes_)
        #             unknown = X_clean[~X_clean.isin(known_classes)].unique()
        #             if len(unknown) > 0:
        #                 raise ValueError(f"Unknown categories in {col}: {unknown}")
        #             X_encoded[f"{col}_encoded"] = encoder.transform(X_clean)
                    
        #         elif self.handle_unknown == 'ignore':
        #             # Map unknown to -1
        #             X_encoded[f"{col}_encoded"] = X_clean.map(
        #                 lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        #             )
                    
        #         # Drop original categorical column
        #         X_encoded = X_encoded.drop(columns=[col])
                
	