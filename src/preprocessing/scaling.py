import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn.pipeline import Pipeline
# eliminar el target column antes del pipeline
# pipeline = Pipeline([
#     ('clean', DataCleaner()),
#     ('features', FeatureEngineer()),
#     ('normalize', DataNormalizer())
# ])

# from sklearn.preprocessing import MaxAbsScaler

# from joblib import Parallel, delayed



class DataScaler(BaseEstimator, TransformerMixin):
	def __init__(self, scaler_type = 'minmax'):
		self.scaler_type = scaler_type
		# self.target_column = target_column
		# self.scaler = None
		# self.feature_columns_ = None
		# self.fitted_ = False

	def fit(self, X, y = None):
		self.numeric_cols_ = X.select_dtypes(include=np.number).columns.tolist()

		# self.feature_columns_ = [col for col in self.numeric_cols_ if col != self.target_column] 
		
		# if self.target_column in self.numeric_cols_:
		# 	self.numeric_cols_ = self.numeric_cols_.drop(self.target_column)

		if self.scaler_type == 'minmax':
			self.scaler_ = MinMaxScaler()
		elif self.scaler_type == 'standard':
			self.scaler_ = StandardScaler()
		elif self.scaler_type == 'robust':
			self.scaler_ = RobustScaler()
		else:
			raise ValueError(f"Scaler unknown: {self.scaler_type}")

		if len(self.numeric_cols_) > 0:
				
			self.scaler_.fit(X[self.numeric_cols_])
			# self.feature_columns_ = self.numeric_cols_
			# self.fitted_ = True
			print(f"Scaler fitted on {len(self.numeric_cols_)} numeric features")
		else:	
			print("No numeric features to scale")	
		return self
	
	def transform(self, X):
		X_scaled = X.copy()

		if self.scaler_ is not None:
			X_scaled[self.numeric_cols_] = self.scaler_.transform(X_scaled[self.numeric_cols_])

		return X_scaled

		# if self.fitted_ and len(self.feature_columns_) > 0 and self.scaler:
		# 	X_scaled[self.feature_columns_] = self.scaler.transform(X[self.feature_columns_])

		# if self.target_column in X.columns:
		# 	X_scaled[self.target_column] = X[self.target_column].apply(
		# 		lambda x: 1 if x != 'BENIGN' else 0
		# 	)

	
class MultiClassLabelEncoder(BaseEstimator, TransformerMixin):

	def __init__(self, target_col='label', benign_value='BENIGN'):
		self.target_col = target_col
		# self.benign_value = benign_value
		# self.classes_ = None
        
	def fit(self, X, y=None):

		if isinstance(X, pd.DataFrame):
			X = X.squeeze()
		
		if self.target_col not in X.columns:
			raise ValueError(f"target column {self.target_col} not found in dataset")

		self.classes_ = sorted(X[self.target_col].astype(str).unique())

		self.class_to_index_ = {
			label: idx for idx, label  in enumerate(self.classes_)
		}

		print(f"Detected {len(self.classes_)} classes")
		for index, value in self.class_to_index_():
			print(f"	{index} -->	{value}") 

		return self
		
	def transform(self, X):
		if not self.fitted_:
			raise ValueError("Label encoder must be fitted before transform")

		X_transformed = X.copy()

		if isinstance(X_transformed, pd.DataFrame):
			X_transformed = X_transformed.squeeze()
		
		if self.target_col in X_transformed.columns:
			X_transformed[self.target_col] = (
				X_transformed[self.target_col].astype(str).map(self.class_to_index_)
			)
					
		if X_transformed[self.target_col].isna().any():
			unseen = X.loc[
				X_transformed[self.target_col].isna(),
				self.target_col
			].unique()
			raise ValueError(f"unseen labels found: {unseen}")
		return X_transformed



if __name__ == "__main__":
    df = pd.read_csv('data/processed/features.csv')
    normalizer = DataScaler(scaler_type='standard')
    normalized_df = normalizer.transform(df)
    normalized_df.to_csv('data/processed/normalized.csv', index=False)

