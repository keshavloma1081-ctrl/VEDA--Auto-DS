"""
VEDA — Autonomous Data Science System
agents/rag/uplift_model.py — Uplift Model Agent
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent


class UpliftModelAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="UpliftModelAgent",
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

    def _simulate_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        np.random.seed(42)
        df = df.copy()
        df["treatment"] = np.random.binomial(1, 0.5, len(df))
        return df

    def _s_learner(self, X: np.ndarray, y: np.ndarray,
                   treatment: np.ndarray) -> dict:
        X_with_t = np.column_stack([X, treatment])
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X_with_t, y, treatment, test_size=0.2, random_state=42
        )
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        X_test_treated = X_test.copy()
        X_test_treated[:, -1] = 1
        X_test_control = X_test.copy()
        X_test_control[:, -1] = 0

        proba_treated = model.predict_proba(X_test_treated)[:, 1]
        proba_control = model.predict_proba(X_test_control)[:, 1]
        cate = proba_treated - proba_control

        return {
            "method": "S-Learner",
            "avg_cate": round(float(cate.mean()), 6),
            "std_cate": round(float(cate.std()), 6),
            "positive_uplift_pct": round(float((cate > 0).mean() * 100), 2)
        }

    def _t_learner(self, X: np.ndarray, y: np.ndarray,
                   treatment: np.ndarray) -> dict:
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, treatment, test_size=0.2, random_state=42
        )
        treated_idx = t_train == 1
        control_idx = t_train == 0

        if treated_idx.sum() < 10 or control_idx.sum() < 10:
            return {"method": "T-Learner", "error": "Insufficient samples"}

        model_t = LogisticRegression(max_iter=500, random_state=42)
        model_c = LogisticRegression(max_iter=500, random_state=42)
        model_t.fit(X_train[treated_idx], y_train[treated_idx])
        model_c.fit(X_train[control_idx], y_train[control_idx])

        proba_t = model_t.predict_proba(X_test)[:, 1]
        proba_c = model_c.predict_proba(X_test)[:, 1]
        cate = proba_t - proba_c

        return {
            "method": "T-Learner",
            "avg_cate": round(float(cate.mean()), 6),
            "std_cate": round(float(cate.std()), 6),
            "positive_uplift_pct": round(float((cate > 0).mean() * 100), 2)
        }

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features for uplift modeling...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        self.log("Simulating treatment assignment...")
        df = self._simulate_treatment(df)

        feature_cols = [c for c in df.columns if c not in [target_col, "treatment"]]

        df_enc = self._encode_df(df[feature_cols])
        X = df_enc.fillna(0).values.astype(np.float32)

        y_raw = df[target_col].values
        if y_raw.dtype == object or str(y_raw.dtype) == "bool":
            y_raw = LabelEncoder().fit_transform(y_raw.astype(str))
        y = y_raw.astype(np.float32)

        treatment = df["treatment"].values

        self.log("Shape: " + str(X.shape))

        self.log("Training S-Learner...")
        s_results = self._s_learner(X, y, treatment)
        self.log("S-Learner CATE: " + str(s_results["avg_cate"]))

        self.log("Training T-Learner...")
        t_results = self._t_learner(X, y, treatment)
        self.log("T-Learner CATE: " + str(t_results.get("avg_cate", "N/A")))

        uplift_results = {
            "s_learner": s_results,
            "t_learner": t_results,
            "treatment_rate": round(float(treatment.mean()), 4),
            "outcome_rate": round(float(y.mean()), 4)
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_uplift_results.json"
        with open(path, "w") as f:
            json.dump(uplift_results, f, indent=2)

        state["uplift_results"] = uplift_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] UpliftModelAgent: " +
            "S-CATE=" + str(s_results["avg_cate"]) +
            " positive_uplift=" + str(s_results["positive_uplift_pct"]) + "%"
        )

        self.log("=" * 50)
        self.log("UPLIFT MODEL COMPLETE")
        self.log("S-Learner CATE : " + str(s_results["avg_cate"]))
        self.log("T-Learner CATE : " + str(t_results.get("avg_cate", "N/A")))
        self.log("Positive uplift: " + str(s_results["positive_uplift_pct"]) + "%")
        self.log("=" * 50)

        return state