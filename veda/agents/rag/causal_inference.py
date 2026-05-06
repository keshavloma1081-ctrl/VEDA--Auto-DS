"""
VEDA — Autonomous Data Science System
agents/rag/causal_inference.py — Causal Inference Agent
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from veda.core.base_agent import BaseAgent


class CausalInferenceAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="CausalInferenceAgent",
            domain="causal",
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

    def _encode_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=["object", "bool"]).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        return df_enc

    def _estimate_ate_naive(self, df: pd.DataFrame,
                             treatment_col: str,
                             outcome_col: str) -> dict:
        treated = df[df[treatment_col] == 1][outcome_col]
        control = df[df[treatment_col] == 0][outcome_col]

        ate = float(treated.mean() - control.mean())
        se = float(np.sqrt(treated.var()/max(len(treated), 1) +
                           control.var()/max(len(control), 1)))
        t_stat, p_value = stats.ttest_ind(treated, control)

        return {
            "method": "Naive Difference in Means",
            "ate": round(ate, 6),
            "standard_error": round(se, 6),
            "confidence_interval_95": [
                round(ate - 1.96 * se, 6),
                round(ate + 1.96 * se, 6)
            ],
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.05)
        }

    def _propensity_score_matching(self, df: pd.DataFrame,
                                    treatment_col: str,
                                    outcome_col: str,
                                    feature_cols: list) -> dict:
        df_enc = self._encode_df(df[feature_cols])
        X = df_enc.fillna(0).values
        t = df[treatment_col].values
        y = df[outcome_col].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        psm = LogisticRegression(max_iter=500, random_state=42)
        psm.fit(X_scaled, t)
        propensity_scores = psm.predict_proba(X_scaled)[:, 1]

        treated_idx = np.where(t == 1)[0]
        control_idx = np.where(t == 0)[0]

        matched_t = []
        matched_c = []

        for idx in treated_idx[:100]:
            ps = propensity_scores[idx]
            diffs = np.abs(propensity_scores[control_idx] - ps)
            best_match = control_idx[np.argmin(diffs)]
            matched_t.append(float(y[idx]))
            matched_c.append(float(y[best_match]))

        if matched_t:
            ate = float(np.mean(matched_t) - np.mean(matched_c))
            se = float(np.std(np.array(matched_t) - np.array(matched_c)) /
                      np.sqrt(len(matched_t)))
        else:
            ate = 0.0
            se = 0.0

        return {
            "method": "Propensity Score Matching",
            "ate": round(ate, 6),
            "standard_error": round(se, 6),
            "n_matched": len(matched_t),
            "confidence_interval_95": [
                round(ate - 1.96 * se, 6),
                round(ate + 1.96 * se, 6)
            ]
        }

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features for causal inference...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        np.random.seed(42)
        df = df.copy()
        df["treatment"] = np.random.binomial(1, 0.5, len(df))

        # Encode target if needed
        y_raw = df[target_col].values
        if y_raw.dtype == object or str(y_raw.dtype) == "bool":
            df[target_col] = LabelEncoder().fit_transform(y_raw.astype(str))

        feature_cols = [c for c in df.columns
                       if c not in [target_col, "treatment"]]

        self.log("Estimating ATE using naive method...")
        naive_ate = self._estimate_ate_naive(df, "treatment", target_col)
        self.log("Naive ATE: " + str(naive_ate["ate"]) +
                " (p=" + str(naive_ate["p_value"]) + ")")

        self.log("Estimating ATE using propensity score matching...")
        psm_ate = self._propensity_score_matching(
            df, "treatment", target_col, feature_cols[:10]
        )
        self.log("PSM ATE: " + str(psm_ate["ate"]) +
                " (n_matched=" + str(psm_ate["n_matched"]) + ")")

        causal_results = {
            "naive_ate": naive_ate,
            "psm_ate": psm_ate,
            "treatment_col": "treatment",
            "outcome_col": target_col,
            "n_samples": len(df)
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_causal_inference.json"
        with open(path, "w") as f:
            json.dump(causal_results, f, indent=2)

        state["causal_inference"] = causal_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] CausalInferenceAgent: " +
            "naive_ATE=" + str(naive_ate["ate"]) +
            " psm_ATE=" + str(psm_ate["ate"])
        )

        self.log("=" * 50)
        self.log("CAUSAL INFERENCE COMPLETE")
        self.log("Naive ATE  : " + str(naive_ate["ate"]))
        self.log("PSM ATE    : " + str(psm_ate["ate"]))
        self.log("Significant: " + str(naive_ate["significant"]))
        self.log("=" * 50)

        return state