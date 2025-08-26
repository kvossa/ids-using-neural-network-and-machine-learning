import re
import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from fuzzywuzzy import process
from normalize import DataNormalizer, LabelBinarizer

# Esto es con Cross-Dataset Feature Harmonization Pipeline

class FeatureHarmonizer(BaseEstimator, TransformerMixin):
	def __init__(self, reference_features):
		self.feature_columns = None
		self.feature_rules = {
			r'time$|timestamp|date': 'timestamp',
		}
		self.reference_features = reference_features
		self.mapping_ = {} 

	def _normalize_name(self, name):
		return re.sub(r'[^a-zA-Z0-9]', '', str(name).lower())
	
	def fit(self, X, y=None):
		for col in X.columns:
			norm_col = self._normalize_name(col)
			match, score = process.extractOne(
				norm_col,
				[self._normalize_name(x) for x in self.reference_features]
			)

			if score > 80:
				matched_feature = self.reference_features[
					[self._normalize_name(x) for x in self.reference_features].index(match)
				]

				self.mapping_[col] = matched_feature
			else:
				self.mapping_[col] = f"raw_{col}"
		return self

	def transform(self, X):
		return X.rename(columns=self.mapping_)
	

class FeatureTypeDetector(BaseEstimator, TransformerMixin):
	def __init__(self, cat_threshold = 0.5):
		self.cat_threshold = cat_threshold
		self.numeric_features_ = []
		self.categorical_features_ = []

	def fit(self, X, y=None):
		for col in X.columns:
			if pd.api.types.is_numeric_dtype(X[col]):
				unique_ratio = X[col].nunique() / len(X[col])
				if 1 < unique_ratio <= self.cat_threshold:
					self.categorical_features_.append(col)
				else:
					self.numeric_features_.append(col)
			
			else:
				self.categorical_features_.append(col)
		return self
	
	def transform(self, X):
		return X

def build_pipeline(target_col='label'):
	harmonizer = FeatureHarmonizer(
		reference_features=[
            'duration', 'protocol', 'src_bytes', 'dst_bytes', 
            'src_ip', 'dst_ip', 'src_port', 'dst_port',
            'packet_count', 'byte_count', 'attack_type'
        ]
	)
	type_detector = FeatureTypeDetector(cat_threshold=0.05)

	preprocessor = ColumnTransformer([
		('num', 'passthrough', []),
		('cat', OneHotEncoder(handle_unknown='ignore'), type_detector.categorical_features_)
	])

	selector = SelectFromModel(
		RandomForestClassifier(n_estimators=50),
		threshold='median'
	)

	normalizer = DataNormalizer(scaler_type='standard')

	binarizer = LabelBinarizer()

	return Pipeline([
		('harmonize', harmonizer),
		('detect_types', type_detector),
		('preprocess', preprocessor),
		('normalize', normalizer),
		('binarize_labels', binarizer),
		('feature_selection', selector),
	])

if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned.csv')
    engineer = FeatureHarmonizer()
    feature_df = engineer.transform(df)
    feature_df.to_csv('data/processed/features.csv', index=False)