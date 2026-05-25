import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.preprocessing.pipeline.features_extraction import FeatureExtraction
from src.preprocessing.pipeline.features_selection import FeatureSelector
from src.preprocessing.pipeline.boruta_shap_selector import HybridFeatureSelector
from src.preprocessing.pipeline.clean import DataCleaner
from src.preprocessing.pipeline.scaling import DataScaler, MultiClassLabelEncoder
from src.preprocessing.pipeline.encoding import CategoricalEncoder
from src.preprocessing.pipeline.imputer import DataFrameImputer


class IDSPipeline:
    def __init__(
        self,
        use_feature_selection=True,
        k_features=30,
        random_state=42,
        dataset="CIC",
        selector_type="hybrid",
        manual_features=None,
    ):
        self.use_feature_selection = use_feature_selection
        self.k_features = k_features
        self.random_state = random_state
        self.dataset = dataset
        self.selector_type = selector_type
        self.manual_features = manual_features
        self.pipeline = None

    def build_pipeline(self):

        steps = [
            ("cleaner", DataCleaner()),
            ("feature_extraction", FeatureExtraction(dataset=self.dataset)),
            ("categorical_encoder", CategoricalEncoder(handle_unknown="ignore")),
            ("imputer", DataFrameImputer(strategy="median")),
            ("robust_scaler", DataScaler(scaler_type="robust")),
            ("minmax_scaler", DataScaler(scaler_type="minmax")),
        ]

        if self.use_feature_selection:
            if self.selector_type == "hybrid":
                steps.append(
                    (
                        "feature_selection",
                        HybridFeatureSelector(
                            dataset=self.dataset,
                            n_trials=100,
                            random_state=self.random_state,
                            use_rfe_tuning=True,
                            verbose=True,
                        ),
                    )
                )
            elif self.selector_type == "boruta":
                from src.preprocessing.pipeline.boruta_shap_selector import (
                    BorutaSHAPSelector,
                )

                steps.append(
                    (
                        "feature_selection",
                        BorutaSHAPSelector(
                            n_trials=100,
                            random_state=self.random_state,
                            verbose=True,
                        ),
                    )
                )
            elif self.selector_type == "rfe":
                from src.preprocessing.pipeline.boruta_shap_selector import (
                    ShapRFESelector,
                )

                steps.append(
                    (
                        "feature_selection",
                        ShapRFESelector(
                            cv=5,
                            scoring="f1_weighted",
                            random_state=self.random_state,
                        ),
                    )
                )
            elif self.selector_type == "fixed":
                from src.preprocessing.pipeline.boruta_shap_selector import (
                    FixedFeatureSelector,
                )

                steps.append(
                    (
                        "feature_selection",
                        FixedFeatureSelector(dataset=self.dataset),
                    )
                )
            elif self.selector_type == "manual_fixed":
                from src.preprocessing.pipeline.boruta_shap_selector import (
                    ManualFeatureSelector,
                )

                steps.append(
                    (
                        "feature_selection",
                        ManualFeatureSelector(
                            dataset=self.dataset,
                            features=self.manual_features,
                            strict=False,
                        ),
                    )
                )
            elif self.selector_type == "manual_hybrid":
                from src.preprocessing.pipeline.boruta_shap_selector import (
                    ManualHybridFeatureSelector,
                )

                steps.append(
                    (
                        "feature_selection",
                        ManualHybridFeatureSelector(
                            dataset=self.dataset,
                            features=self.manual_features,
                            k_features=self.k_features,
                            random_state=self.random_state,
                        ),
                    )
                )
            else:
                steps.append(
                    (
                        "feature_selection",
                        FeatureSelector(
                            k_features=self.k_features, random_state=self.random_state
                        ),
                    )
                )

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
