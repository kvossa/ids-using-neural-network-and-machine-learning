from load import DatasetProcessor
from scaling import DataNormalizer, LabelBinarizer
from pathlib import Path
from encoding import CategoricalEncoder
from split import split_and_store
import json
import glob
from sklearn.pipeline import Pipeline

def main():
	datasets = {
		'UNSW-NB15': Path('data/raw/UNSW-NB15/UNSW-NB15_training-set.csv'),
		'CIC-IDS2017': [Path(p) for p in glob.glob('data/raw/CIC-IDS2017/*.parquet')],
		# 'CIC-IDS-Collection': Path('data/raw/cic-collection.parquet'),
		# 'CSE-CIC-IDS2018': [Path(p) for p in glob.glob('data/raw/CSE-CIC-IDS2018/*.csv')],
		# 'CIC-IDS-2017-PCAP': [Path(p) for p in glob.glob('data/raw/Network Intrusion dataset(CIC-IDS-2017)/*.csv')],
	}

	processed_data = {}
	metadata = {}

	for name, path in datasets.items():
		print(f"Processing {name}")
		processor = DatasetProcessor
		features, labels = processor.process(path)

		processed_data[name] = {
			'features': features,
			'labels': labels
		}

		metadata[name] = processor.feature_mapping_

		output_dir = Path('../data/processed') / name
		output_dir.mkdir(parents=True, exist_ok=True)

		features.to_parquet(output_dir / 'features.parquet')
		labels.to_csv(output_dir / 'labels.csv')

		with open (output_dir / 'features_mapping.json') as f:
			json.dump(processor.feature_mapping_, f)

		split_and_store(
			features,
			output_dir,
			test_size=0.2,
			val_size=0.1,
			random_state=42,
			stratify_col='label'
		)


	print('processing complete')


if __name__ == '__main__':
	main() 

