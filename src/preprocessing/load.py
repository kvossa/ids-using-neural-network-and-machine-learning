import pandas as pd
from pathlib import Path
from features import build_pipeline
# Load with pd.read_parquet(), requires fastparquet or pyarrow

class DatasetProcessor:
	def __init__(self, dataset_name: str):
		self.dataset_name = dataset_name.lower()
		self.pipeline = build_pipeline()
		self.feature_mapping_ = {}


	def load_raw_datasets(self, source: str) -> pd.DataFrame:
		if self.dataset_name == 'unsw-nb15':
			df = pd.read_csv(source)
			# df = self._process_unsw(df)
		elif self.dataset_name == 'cic-ids2017':
			df = pd.read_parquet(source)
		return df
	
	def _process_unsw_columns(self, df):
		return
	
	def process(self, source):
		raw_df = self.load_raw_datasets(source)
		processed = self.pipeline.fit_transform(raw_df.drop(columns=['label']), raw_df['label'])

		self.feature_mapping_ = {
			'original_columns': list(raw_df.columns),
			'final_features': self.pipeline.named_steps['harmonize'].mapping_
		}

		return pd.DataFrame(processed), raw_df['label']



# if __name__ == '__main__':
	# pass
	# cic_2018 = load_datasets('CSE-CIC-IDS2018', '2017')
	# cic_