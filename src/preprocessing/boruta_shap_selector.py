import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier


LEAKY_FEATURES = {
    "UNSW": [
        "Stime",
        "Ltime",
        "srcip",
        "dstip",
        "ct_srv_src", #
        "ct_srv_dst", #
        "ct_dst_ltm", #
        "ct_src_ltm", #
        "ct_src_dport_ltm", #
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "ct_flw_http_mthd", #
        "ct_ftp_cmd",
        "ct_state_ttl",
        "is_ftp_login", #
        "is_sm_ips_ports",
    ],
    "CIC": [
        "Timestamp",
        "Source IP",
        "Destination IP",
        "flow_solver",
        "Src Upper Estimate",
        "Dst Upper Estimate",
    ],
}

REDUNDANT_FEATURES = {
    "UNSW": {
        "Sload": ["sbytes", "dur"],
        "Dload": ["dbytes", "dur"],
        "smeansz": ["sbytes", "spkts"],
        "dmeansz": ["dbytes", "dpkts"],
        "tcprtt": ["synack", "ackdat"],
        "swin": ["sbytes"],
        "dwin": ["dbytes"],
        "stcpb": ["sbytes"],
        "dtcpb": ["dbytes"],
    },
    "CIC": {},
}


class FeaturePreFilter(BaseEstimator, TransformerMixin):
    def __init__(self, dataset="UNSW", remove_leaky=True, remove_redundant=True):
        self.dataset = dataset.upper()
        self.remove_leaky = remove_leaky
        self.remove_redundant = remove_redundant
        self.removed_features_ = []

    def fit(self, X, y=None):
        self.removed_features_ = []
        to_remove = set()

        if self.remove_leaky:
            leaky = set(LEAKY_FEATURES.get(self.dataset, []))
            existing_leaky = leaky.intersection(set(X.columns))
            to_remove.update(existing_leaky)
            self.removed_features_.append(("leaky", list(existing_leaky)))

        if self.remove_redundant:
            redundant_map = REDUNDANT_FEATURES.get(self.dataset, {})
            for feat, deps in redundant_map.items():
                if feat in X.columns:
                    deps_exist = all(d in X.columns for d in deps)
                    if deps_exist:
                        to_remove.add(feat)
            self.removed_features_.append(("redundant", list(to_remove)))

        self.to_remove_ = list(to_remove)
        self.kept_features_ = [c for c in X.columns if c not in to_remove]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_remove_, errors="ignore")

    def get_report(self):
        return {
            "removed": self.to_remove_,
            "kept": self.kept_features_,
            "details": dict(self.removed_features_),
        }


class BorutaSHAPSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_trials=100,
        random_state=42,
        sample=True,
        sample_fraction=0.1,
        importance_measure="shap",
        percentile=100,
        model_type="lightgbm",
        verbose=True,
    ):
        self.n_trials = n_trials
        self.random_state = random_state
        self.sample = sample
        self.sample_fraction = sample_fraction
        self.importance_measure = importance_measure
        self.percentile = percentile
        self.model_type = model_type
        self.verbose = verbose
        self.selector_ = None
        self.accepted_ = []
        self.rejected_ = []
        self.tentative_ = []

    def _get_model(self):
        if self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                verbose=-1,
            )
        elif self.model_type == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                verbosity=0,
            )
        else:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1,
            )

    def fit(self, X, y):
        # Use basic boruta package instead of BorutaShap
        try:
            from boruta import BorutaPy
        except ImportError:
            raise ImportError("boruta not installed. Run: pip install boruta")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names_ = X.columns.tolist()
        else:
            X_array = X
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        model = self._get_model()

        self.selector_ = BorutaPy(
            model,
            n_estimators='auto',
            max_iter=self.n_trials,
            random_state=self.random_state,
            verbose=self.verbose if self.verbose else 0,
        )

        self.selector_.fit(X_array, y)

        self.accepted_ = [
            self.feature_names_[i]
            for i, selected in enumerate(self.selector_.support_)
            if selected
        ]
        self.tentative_ = [
            self.feature_names_[i]
            for i, selected in enumerate(self.selector_.support_weak_)
            if selected
        ]
        self.rejected_ = [
            f for f in self.feature_names_
            if f not in self.accepted_ and f not in self.tentative_
        ]

        if self.verbose:
            print(f"\n[Boruta] Accepted: {len(self.accepted_)} features")
            print(f"[Boruta] Tentative: {len(self.tentative_)} features")
            print(f"[Boruta] Rejected: {len(self.rejected_)} features")

        self.selected_features_ = self.accepted_ + self.tentative_

        return self

    def transform(self, X):
        if self.selector_ is None:
            raise ValueError("Call fit() before transform()")

        selected = self.accepted_ + self.tentative_
        if isinstance(X, pd.DataFrame):
            return X[selected]
        else:
            indices = [self.feature_names_.index(f) for f in selected]
            return X[:, indices]

    def get_selected_features(self):
        return self.accepted_ + self.tentative_

    def get_importance_ranking(self):
        return self.selector_.history


class ShapRFESelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_estimators=50,
        max_depth=5,
        step=1,
        cv=5,
        scoring="f1_weighted",
        min_features_to_select=5,
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.selector_ = None
        self.selected_features_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        try:
            import shap
        except ImportError:
            raise ImportError("shap not installed. Run: pip install shap")

        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        else:
            model = LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
            )

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_fit = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_fit = X

        if self.min_features_to_select is None:
            self.min_features_to_select = max(5, int(X_fit.shape[1] * 0.1))

        self.selector_ = RFECV(
            estimator=model,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.min_features_to_select,
            n_jobs=self.n_jobs,
        )

        self.selector_.fit(X_fit, y)

        selected_mask = self.selector_.get_support()
        self.selected_features_ = [
            f for f, s in zip(self.feature_names_, selected_mask) if s
        ]
        self.n_features_ = self.selector_.n_features_

        if hasattr(self.selector_.estimator_, "feature_importances_"):
            self.feature_importances_ = dict(
                zip(self.feature_names_, self.selector_.estimator_.feature_importances_)
            )

        if self.selector_.cv_results_.get("mean_test_score") is not None:
            scores = self.selector_.cv_results_["mean_test_score"]
            n_features_range = range(
                self.min_features_to_select,
                len(scores) + self.min_features_to_select,
            )
            optimal_idx = np.argmax(scores)
            self.optimal_n_features_ = list(n_features_range)[optimal_idx]
            self.optimal_score_ = scores[optimal_idx]
            print(
                f"[ShapRFESelector] Optimal: {self.optimal_n_features_} features (score: {self.optimal_score_:.4f})"
            )

        return self

    def transform(self, X):
        if self.selector_ is None:
            raise ValueError("Call fit() before transform()")
        return X[self.selected_features_]

    def get_selected_features(self):
        return self.selected_features_


class HybridFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        dataset="UNSW",
        n_trials=100,
        random_state=42,
        use_rfe_tuning=True,
        verbose=True,
    ):
        self.dataset = dataset.upper()
        self.n_trials = n_trials
        self.random_state = random_state
        self.use_rfe_tuning = use_rfe_tuning
        self.verbose = verbose
        self.prefilter_ = None
        self.boruta_ = None
        self.rfe_ = None
        self.selected_features_ = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        if self.verbose:
            print(
                f"\n[HybridFeatureSelector] Starting with {len(self.feature_names_in_)} features"
            )
            print(f"[HybridFeatureSelector] Step 1: Pre-filtering...")

        self.prefilter_ = FeaturePreFilter(dataset=self.dataset)
        X_filtered = self.prefilter_.fit_transform(X)

        if self.verbose:
            report = self.prefilter_.get_report()
            print(
                f"[HybridFeatureSelector] Removed {len(report['removed'])} features: {report['removed']}"
            )
            print(f"[HybridFeatureSelector] Remaining: {len(report['kept'])} features")
            print(f"[HybridFeatureSelector] Step 2: Boruta-SHAP...")

        MAX_SAMPLES = 100_000
        if len(X_filtered) > MAX_SAMPLES:
            from sklearn.model_selection import StratifiedShuffleSplit

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=MAX_SAMPLES,
                random_state=self.random_state,
            )
            _, sample_idx = next(sss.split(X_filtered, y))
            X_boruta = (
                X_filtered.iloc[sample_idx]
                if hasattr(X_filtered, "iloc")
                else X_filtered[sample_idx]
            )
            y_boruta = (
                y[sample_idx] if isinstance(y, np.ndarray) else np.array(y)[sample_idx]
            )
            print(
                f"[HybridFeatureSelector] Subsampling to {MAX_SAMPLES:,} samples for Boruta"
            )
        else:
            X_boruta = X_filtered
            y_boruta = y

        self.boruta_ = BorutaSHAPSelector(
            n_trials=self.n_trials,
            random_state=self.random_state,
            sample=True,
            sample_fraction=0.1,
            verbose=self.verbose,
        )
        self.boruta_.fit(X_boruta, y_boruta)

        boruta_selected = self.boruta_.get_selected_features()

        if self.verbose:
            print(
                f"[HybridFeatureSelector] Boruta selected {len(boruta_selected)} features"
            )

        if self.use_rfe_tuning and len(boruta_selected) > 5:
            if self.verbose:
                print(
                    f"[HybridFeatureSelector] Step 3: RFE fine-tuning on {len(boruta_selected)} features..."
                )

            X_rfe = (
                X_filtered[boruta_selected]
                if isinstance(X_filtered, pd.DataFrame)
                else X_filtered[
                    :, [self.feature_names_in_.index(f) for f in boruta_selected]
                ]
            )

            self.rfe_ = ShapRFESelector(
                min_features_to_select=max(5, len(boruta_selected) // 2),
                cv=5,
                scoring="f1_weighted",
                random_state=self.random_state,
            )
            self.rfe_.fit(X_rfe, y)
            self.selected_features_ = self.rfe_.get_selected_features()
        else:
            self.selected_features_ = boruta_selected

        if self.verbose:
            print(
                f"[HybridFeatureSelector] Final: {len(self.selected_features_)} features selected"
            )
            print(f"[HybridFeatureSelector] Features: {self.selected_features_}")

        return self

    def transform(self, X):
        return X[self.selected_features_]

    def get_selected_features(self):
        return self.selected_features_

    def get_pipeline_summary(self):
        return {
            "prefilter": self.prefilter_.get_report() if self.prefilter_ else None,
            "boruta": {
                "accepted": self.boruta_.accepted_ if self.boruta_ else [],
                "rejected": self.boruta_.rejected_ if self.boruta_ else [],
                "tentative": self.boruta_.tentative_ if self.boruta_ else [],
            },
            "final": self.selected_features_,
        }
