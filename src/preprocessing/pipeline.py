import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from features_extraction import FeatureExtraction
from features_selection import FeatureSelector
from clean import DataCleaner
from scaling import DataScaler, MultiClassLabelEncoder
from encoding import CategoricalEncoder

class IDSPipeline:
	def __init__(self, use_feature_selection=True, k_features=30, random_state=42, dataset="CIC"):
		# self.model_name = model_name
		self.use_feature_selection = use_feature_selection
		self.k_features = k_features
		self.random_state = random_state
		self.dataset = dataset
		self.pipeline = None

	def build_pipeline(self):
		# categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
		# numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
		# preprocessor = ColumnTransformer(
		# 	transformers=[
		# 		("num", StandardScaler(), numeric_cols),
		# 		("cat", CategoricalEncoder(
		# 			categorical_columns=categorical_cols,
		# 			handle_unknown="ignore"
		# 		), categorical_cols), 
		# 	],
		# 	remainder="drop"
		# )

		steps = [
			("cleaner", DataCleaner()),
			("feature_extraction", FeatureExtraction(dataset=self.dataset)),
			# ("encoder", MultiClassLabelEncoder()) Aplicar antes de entrenar el Pipeline
			("categorical_encoder", CategoricalEncoder(handle_unknown='ignore')),
			("scaler", DataScaler()),
			# ("preprocessor", preprocessor)
		]

		if self.use_feature_selection:
			steps.append(("feature_selection", FeatureSelector(k_features=self.k_features, random_state=self.random_state)))
		
		# steps.append(("model", get_model(self.model_name, self.random_state)))

		self.pipeline = Pipeline(steps)

		return self.pipeline

	def fit(self, X_train, y_train):
		self.pipeline.fit(X_train, y_train)

	def transform(self, X):
		return self.pipeline.transform(X)

	def fit_transform(self, X, y=None):
		return self.pipeline.fit_transform(X, y)

	def predict(self, X_test):
		return self.pipeline.predict(X_test)

	def predict_proba(self, X_test):
		return self.pipeline.predict_proba(X_test)

# if __name__ == "__main__":
# 	cic_df = pd.read_parquet('../../data/cic_ids2017.parquet')
# 	unsw_df = pd.read_csv('data/unsw_nb15.csv')
