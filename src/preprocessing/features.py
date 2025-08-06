import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn.feature_selection import VarianceThreshold
# selector = VarianceThreshold(threshold=0.01)

class FeatureEngineer(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.feature_columns = None

	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		if 'flow_duration' in X.columns:
			X['flow_duration_sec'] = X['flow_duration'].apply(lambda x: x / 1e6)

		if 'packet_length_mean' in X.columns:
			X['packet_size_diff'] = X['packet_length_max'] - X['packet_length_min']

		if 'protocol' in X.columns:
			protocol_dummies = pd.get_dummies(X['protocol'], prefix='proto')
			X = pd.concat([X, protocol_dummies], axis=1)

		tcp_flags = ['flag_SYN', 'flag_ACK', 'flag_FIN']
		for flag in tcp_flags:
			if flag in X.columns:
				X[f'{flag}_ratio'] = X[flag] / X['total_packets']
		
		self.feature_columns = X.columns.tolist()
		return X

if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned.csv')
    engineer = FeatureEngineer()
    feature_df = engineer.transform(df)
    feature_df.to_csv('data/processed/features.csv', index=False)