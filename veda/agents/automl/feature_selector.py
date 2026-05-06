"""
VEDA — Autonomous Data Science System
agents/automl/feature_selector.py — Feature Selector Agent

Automated feature selection:
- Variance threshold
- Correlation filter
- Recursive feature elimination
- Importance-based selection
- SHAP-based selection
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier

from veda.core.base_agent import BaseAgent


class FeatureSelectorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="FeatureSelectorAgent",
            domain="automl",
            version="1.0.0"
        )

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _encode_df(self, df):
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=["object", "bool"]).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        return df_enc

    def _variance_threshold(self, X: np.ndarray,
                             feature_names: list,
                             threshold: float = 0.01) -> list:
        """Remove low-variance features."""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        mask = selector.get_support()
        selected = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        self.log("Variance threshold: " + str(len(selected)) +
                "/" + str(len(feature_names)) + " features kept")
        return selected

    def _correlation_filter(self, df: pd.DataFrame,
                             feature_cols: list,
                             threshold: float = 0.95) -> list:
        """Remove highly correlated features."""
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns
                  if any(upper[col] > threshold)]
        selected = [f for f in feature_cols if f not in to_drop]
        self.log("Correlation filter: removed " + str(len(to_drop)) +
                " features, kept " + str(len(selected)))
        return selected

    def _importance_selection(self, X: np.ndarray, y: np.ndarray,
                               feature_names: list,
                               top_k: int = 20) -> list:
        """Select top features by importance."""
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        selected = [feature_names[i] for i in indices]
        importance_dict = {
            feature_names[i]: round(float(importances[i]), 6)
            for i in indices
        }
        self.log("Importance selection: top " + str(len(selected)) + " features")
        return selected, importance_dict

    def _univariate_selection(self, X: np.ndarray, y: np.ndarray,
                               feature_names: list,
                               k: int = 15) -> list:
        """Univariate feature selection using f_classif."""
        k = min(k, X.shape[1])
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        selected = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        self.log("Univariate selection: " + str(len(selected)) + " features")
        return selected

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features for selection...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        feature_cols = [c for c in df.columns if c != target_col]
        df_enc = self._encode_df(df)
        X = df_enc[feature_cols].fillna(0).values.astype(np.float32)
        y_raw = df[target_col].values
        if y_raw.dtype == object:
            y_raw = LabelEncoder().fit_transform(y_raw.astype(str))
        y = y_raw.astype(np.float32)

        self.log("Original features: " + str(len(feature_cols)))

        # Step 1 — Variance threshold
        var_selected = self._variance_threshold(X, feature_cols)

        # Step 2 — Correlation filter
        corr_selected = self._correlation_filter(
            df_enc, var_selected
        )

        # Step 3 — Importance selection
        X_filtered = df_enc[corr_selected].fillna(0).values.astype(np.float32)
        top_k = min(20, len(corr_selected))
        imp_selected, importance_dict = self._importance_selection(
            X_filtered, y, corr_selected, top_k=top_k
        )

        # Step 4 — Univariate selection
        uni_selected = self._univariate_selection(
            X_filtered, y, corr_selected, k=min(15, len(corr_selected))
        )

        # Consensus: features selected by both importance and univariate
        consensus = list(set(imp_selected) & set(uni_selected))
        if not consensus:
            consensus = imp_selected[:10]

        self.log("Final selected features: " + str(len(consensus)))
        self.log("Top features: " + str(consensus[:5]))

        selection_results = {
            "original_features": len(feature_cols),
            "after_variance_threshold": len(var_selected),
            "after_correlation_filter": len(corr_selected),
            "importance_selected": len(imp_selected),
            "univariate_selected": len(uni_selected),
            "final_selected": len(consensus),
            "selected_features": consensus,
            "feature_importances": importance_dict,
            "reduction_pct": round((1 - len(consensus)/len(feature_cols)) * 100, 2)
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_feature_selection.json"
        with open(path, "w") as f:
            json.dump(selection_results, f, indent=2)

        state["feature_selection"] = selection_results
        state["selected_features"] = consensus
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] FeatureSelectorAgent: " +
            str(len(feature_cols)) + " -> " + str(len(consensus)) +
            " features (" + str(selection_results["reduction_pct"]) + "% reduction)"
        )

        self.log("=" * 50)
        self.log("FEATURE SELECTION COMPLETE")
        self.log("Original  : " + str(len(feature_cols)))
        self.log("Selected  : " + str(len(consensus)))
        self.log("Reduction : " + str(selection_results["reduction_pct"]) + "%")
        self.log("=" * 50)

        return state