import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn.pipeline import Pipeline
# pipeline = Pipeline([
#     ('clean', DataCleaner()),
#     ('features', FeatureEngineer()),
#     ('normalize', DataNormalizer())
# ])

# from sklearn.preprocessing import MaxAbsScaler

# from joblib import Parallel, delayed



class DataNormalizer(BaseEstimator, TransformerMixin):
	def __init__(self, scaler_type = 'minmax', target_column = 'label'):
		self.scaler_type = scaler_type
		self.target_column = target_column
		self.scaler = None
		self.feature_columns_ = None
		self.fitted = False

	def fit(self, X, y = None):
		numeric_cols = X.select_dtypes(include=np.number).columns
		
		if self.target_column in numeric_cols:
			numeric_cols = numeric_cols.drop(self.target_column)

		if self.scaler_type == 'minmax':
			self.scaler = MinMaxScaler()
		else:
			self.scaler = StandardScaler()

		if len(self.feature_columns_) > 0:
				
			self.scaler.fit(X[numeric_cols])
			self.feature_columns = numeric_cols
			self.fitted_ = True
		return self
	
	def transform(self, X):
		if not self.fitted_:
			return X

		X_scaled = X.copy()
		
		if self.scaler:
			X.scaled[self.feature_columns] = self.scaler.transform(X[self.feature_columns])

		if self.target_column in X.columns:
			X_scaled[self.target_column] = X[self.target_column].apply(
				lambda x: 1 if x != 'BENIGN' else 0
			)

		return X_scaled
	
class LabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, target_col='label', benign_value='BENIGN'):
        self.target_col = target_col
        self.benign_value = benign_value
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_transformed = X.copy()
        if self.target_col in X.columns:
            X_transformed[self.target_col] = (
                X[self.target_col] != self.benign_value
            ).astype(int)
        return X_transformed
	
if __name__ == "__main__":
    df = pd.read_csv('data/processed/features.csv')
    normalizer = DataNormalizer(scaler_type='standard')
    normalized_df = normalizer.transform(df)
    normalized_df.to_csv('data/processed/normalized.csv', index=False)

