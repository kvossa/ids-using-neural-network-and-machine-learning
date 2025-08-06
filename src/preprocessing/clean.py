import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
	def __init__(self, drop_columns=['flow_id', 'src_ip', 'dst_ip']):
		self.drop_columns = drop_columns

	def fit(self, X, y = None):
		return self
	
	def transform(self, X):
		X_clean = X.drop(columns=self.drop_columns, errors='ignore')
		X_clean = X_clean.drop_duplicates()

		numeric_cols = X_clean.select_dtypes(include=np.number).columns
		X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].median())

		X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

		return X_clean

if __name__ == "__main__":
    df = pd.read_csv('data/raw/cicids2017/MachineLearningCSV.csv')
    cleaner = DataCleaner()
    cleaned_df = cleaner.transform(df)
    cleaned_df.to_csv('data/processed/cleaned.csv', index=False)