from .load import load_datasets
from .split import split_data
from .clean import DataCleaner
from .features import FeatureEngineer
from .normalize import DataNormalizer

if __name__ == '__main__':
	cic_2017 = load_datasets('cic-ids2017')
