# feature_extraction_unsw.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

###### BASE FEATURES

class RatioFeatures(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        if {'sbytes', 'dbytes'}.issubset(X_transformed.columns):
            X_transformed['bytes_ratio'] = X_transformed['sbytes'] / (X_transformed['dbytes'] + 1)
            # X_transformed['total_bytes'] = X_transformed['sbytes'] + X_transformed['dbytes']
        
        if {'spkts', 'dpkts'}.issubset(X_transformed.columns):
            X_transformed['packets_ratio'] = X_transformed['spkts'] / (X_transformed['dpkts'] + 1)
            # X_transformed['total_packets'] = X_transformed['spkts'] + X_transformed['dpkts']

        # if 'total_bytes' in X_transformed.columns:
        #     X_transformed['avg_bytes_per_packet'] = (
        #         X_transformed['total_bytes'] / (X_transformed['total_packets'] + 1)
        #     )

        return X_transformed

class InteractionFeatures(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # if {'ct_srv_src', 'ct_state_ttl'}.issubset(X_transformed.columns):
        #     X_transformed['service_state_interaction'] = (
        #         X_transformed['ct_srv_src'] * X_transformed['ct_state_ttl']
        #     )

        return X_transformed

class RateFeatures(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # if {'sbytes', 'dur'}.issubset(X_transformed.columns):
        #     X_transformed['src_bytes_per_second'] = X_transformed['sbytes'] / (X_transformed['dur'] + 0.001)

        # if {'dbytes', 'dur'}.issubset(X_transformed.columns):
        #     X_transformed['dst_bytes_per_second'] = X_transformed['dbytes'] / (X_transformed['dur'] + 0.001)

        if {'spkts', 'dur'}.issubset(X_transformed.columns):
            X_transformed['packets_per_second'] = X_transformed['spkts'] / (X_transformed['dur'] + 0.001)

        return X_transformed

###### DATASET SPECIFIC 

class UNSWFeatures(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        if 'sttl' in X_transformed.columns and 'dttl' in X_transformed.columns:
            X_transformed['ttl_difference'] = abs(X_transformed['sttl'] - X_transformed['dttl'])
        
        return X_transformed

class CICFeatures(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        if "Fwd Seg Size Min" in X_transformed.columns:
            X_transformed["Fwd Seg Size Min"] = X_transformed["Fwd Seg Size Min"].clip(lower=0)

        X_transformed['fwd_packets_per_second'] = (
            X_transformed['Total Fwd Packets'] / ((X_transformed['Flow Duration']/1_000_000) + 1e-6)
        )
        
        return X_transformed

###### MASTER

class FeatureExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        self.dataset = dataset.upper()
        self.common_blocks = [
            RatioFeatures(),
            RateFeatures(),
            InteractionFeatures()
        ]
        self.specific_blocks = {
            "CIC": CICFeatures(),
            "UNSW": UNSWFeatures()
        } 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        for block in self.common_blocks:
            X_transformed = block.transform(X_transformed)

        if self.dataset in self.specific_blocks:
            X_transformed = self.specific_blocks[self.dataset].transform(X_transformed)

        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)

        print(f"Feature extraction completed for {self.dataset}")
        print(f"total features: {len(X_transformed.columns)}")

        return X_transformed