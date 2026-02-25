import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, RFE, mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier


class FeatureSelector: 

    def __init__(self, k_features=20, random_state=42, use_anova=False, use_mi=True, use_rf=False, use_rfe=False):
        self.k_features = k_features
        self.random_state = random_state
        self.use_anova = use_anova
        self.use_mi = use_mi
        self.use_rf = use_rf
        self.use_rfe = use_rfe
        self.results_ = None
        self.selected_features_ = None

    def fit(self, X, y):
        self.feature_names_ = X.columns.tolist()
        importance_df = pd.DataFrame({"feature": self.feature_names_})

        if self.use_anova:
            f_selector = SelectKBest(score_func=f_classif, k="all")
            f_selector.fit(X, y)
            importance_df["f_score"] = f_selector.scores_
        
        if self.use_mi:
            mi_selector = SelectKBest(score_func=mutual_info_classif, k="all")
            mi_selector.fit(X, y)
            importance_df["mi_score"] = mi_selector.scores_

        if self.use_rf:
            rf = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=1)
            rf.fit(X, y)
            importance_df["rf_importance"] = rf.feature_importances_

        if self.use_rfe:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1)
            rfe = RFE(estimator=estimator, n_features_to_select=self.k_features)            
            rfe.fit(X, y)
            importance_df["rfe_rank"] = rfe.ranking_
        
        score_columns = [col for col in importance_df.columns if col != "feature"]

        for col in score_columns:
            col_min = importance_df[col].min()
            col_max = importance_df[col].max()

            if col_max - col_min == 0:
                importance_df[col + "_norm"] = 0
            else:
                importance_df[col + "_norm"] = (
                    (importance_df[col] - col_min) / (col_max - col_min)
                )
            
        if "rfe_rank_norm" in importance_df.columns:
            importance_df["rfe_rank_norm"] = 1 - importance_df["rfe_rank_norm"]

        norm_cols = [col for col in importance_df.columns if col.endswith("_norm")]

        importance_df["consensus"] = (importance_df[norm_cols].mean(axis=1))

        importance_df = importance_df.sort_values("consensus", ascending=False)

        self.results_ = importance_df

        self.selected_features_ = (importance_df["feature"].head(self.k_features).tolist())

        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("Call fit() before translation")
        
        return X[self.selected_features_]

        
        # feature_index = [self.feature_names_.index(f) for f in self.selected_features_]

        # return X[:, feature_index]

    def get_feature_ranking(self):
        return self.results_

    def get_selected_features(self):
        return self.selected_features_

    