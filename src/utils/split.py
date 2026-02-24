from sklearn.model_selection import train_test_split
import pandas as pd
import json
from pathlib import Path


def split_and_store(
		df: pd.DataFrame,
		output_dir: Path,
		test_size: float = 0.2,
		val_size: float = 0.1,
		random_state: int = 42,
		stratify_col: str = 'attack_type'
) -> None:
	
	(output_dir / 'train').mkdir(parents=True, exist_ok=True)
	(output_dir / 'test').mkdir(parents=True, exist_ok=True)
	(output_dir / 'val').mkdir(parents=True, exist_ok=True)

	train, test = train_test_split(
		df,
		test_size = test_size,
		stratify = df[stratify_col],
		random_state = random_state,
	)

	train, val = train_test_split(
		train,
		test_size = val_size / (1 - test_size),
		stratify = train[stratify_col],
		random_state = random_state,
	)

	splits = {
		'train': train,
		'test': test,
		'val': val
	}

	for split_name, split_df in splits.items():
		split_df.to_parquet(output_dir / split_name / 'data.parquet')

		metadata = {
			'num_samples': len(split_df),
			'class_distribution': split_df[stratify_col].value_counts(normalize = True).to_dict(),
			'split_parameters': {
				'test_size': test_size,
				'val_size': val_size,
				'random_state': random_state,
			} 
		}

		with open(output_dir / split_name / 'metadata.json', 'w') as f:
			json.dump(metadata, f, indent = 2)
		
	return {
		'train': output_dir / 'train' / 'data.parquet',
		'test': output_dir / 'test' / 'data.parquet',
		'val': output_dir / 'val' / 'data.parquet'
	}


if __name__ == "__main__":
	dataframe = pd.read_parquet('../../data/processed/CIC-IDS2017/cic_assembled.parquet')
	output_folder = '../../data/processed/CIC-IDS2017/splits'
	output_path = Path(output_folder)
	split_and_store(df=dataframe, output_dir=output_path)
	