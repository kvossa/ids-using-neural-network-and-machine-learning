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
		self.feature_columns = None

	def fit(self, X, y = None):
		numeric_cols = X.select_dtypes(include=np.number).columns
		
		if self.target_column in numeric_cols:
			numeric_cols = numeric_cols.drop(self.target_column)

		if self.scaler_type == 'minmax':
			self.scaler = MinMaxScaler()
		else:
			self.scaler = StandardScaler()

		self.scaler.fit(X[numeric_cols])
		self.feature_columns = numeric_cols
		return self
	
	def transform(self, X):
		X_scaled = X.copy()
		
		if self.scaler:
			X.scaled[self.feature_columns] = self.scaler.transform(X[self.feature_columns])

		if self.target_column in X.columns:
			X_scaled[self.target_column] = X[self.target_column].apply(
				lambda x: 1 if x != 'BENIGN' else 0
			)

		return X_scaled
	
if __name__ == "__main__":
    df = pd.read_csv('data/processed/features.csv')
    normalizer = DataNormalizer(scaler_type='standard')
    normalized_df = normalizer.transform(df)
    normalized_df.to_csv('data/processed/normalized.csv', index=False)

