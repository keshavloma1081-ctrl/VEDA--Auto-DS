"""
VEDA — Autonomous Data Science System
agents/automl/automl_search.py — AutoML Search Agent

Automated model search using FLAML:
- Auto model selection
- Auto hyperparameter tuning
- Time-budget based search
- Best model reporting
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from veda.core.base_agent import BaseAgent


class AutoMLSearchAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="AutoMLSearchAgent",
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

    def _run_flaml(self, X_train, y_train, X_test, y_test,
                   time_budget: int = 30) -> dict:
        """Run FLAML AutoML search."""
        try:
            from flaml import AutoML
            automl = AutoML()
            automl.fit(
                X_train, y_train,
                task="classification",
                time_budget=time_budget,
                metric="roc_auc",
                verbose=0
            )

            y_pred_proba = automl.predict_proba(X_test)[:, 1]
            y_pred = automl.predict(X_test)

            auc = round(float(roc_auc_score(y_test, y_pred_proba)), 4)
            f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
            acc = round(float(accuracy_score(y_test, y_pred)), 4)

            return {
                "best_estimator": str(automl.best_estimator),
                "best_config": str(automl.best_config)[:200],
                "best_loss": round(float(automl.best_loss), 6),
                "auc": auc,
                "f1": f1,
                "accuracy": acc,
                "time_budget": time_budget,
                "status": "success"
            }
        except Exception as e:
            self.log("FLAML failed: " + str(e), level="WARN")
            return {"status": "failed", "error": str(e)}

    def _run_manual_search(self, X_train, y_train,
                            X_test, y_test) -> dict:
        """Manual model search as fallback."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        import lightgbm as lgb

        models = {
            "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42)
        }

        results = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                auc = round(float(roc_auc_score(y_test, y_pred_proba)), 4)
                f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
                results[name] = {"auc": auc, "f1": f1}
                self.log(name + " AUC=" + str(auc))
            except Exception as e:
                self.log(name + " failed: " + str(e), level="WARN")

        if results:
            best = max(results, key=lambda k: results[k]["auc"])
            return {
                "best_estimator": best,
                "all_results": results,
                "auc": results[best]["auc"],
                "f1": results[best]["f1"],
                "status": "success"
            }
        return {"status": "failed"}

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features for AutoML...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        df_enc = self._encode_df(df)
        X = df_enc.drop(columns=[target_col]).fillna(0).values.astype(np.float32)
        y_raw = df[target_col].values
        if y_raw.dtype == object:
            y_raw = LabelEncoder().fit_transform(y_raw.astype(str))
        y = y_raw.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.log("Train: " + str(len(X_train)) + " Test: " + str(len(X_test)))

        # Try FLAML first
        self.log("Running FLAML AutoML search (30s budget)...")
        results = self._run_flaml(X_train, y_train, X_test, y_test, time_budget=30)

        if results.get("status") == "failed":
            self.log("FLAML unavailable — running manual search...")
            results = self._run_manual_search(X_train, y_train, X_test, y_test)

        self.log("Best estimator: " + str(results.get("best_estimator")))
        self.log("AUC: " + str(results.get("auc")))

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_automl_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        state["automl_results"] = results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] AutoMLSearchAgent: " +
            "best=" + str(results.get("best_estimator")) +
            " AUC=" + str(results.get("auc"))
        )

        self.log("=" * 50)
        self.log("AUTOML SEARCH COMPLETE")
        self.log("Best model : " + str(results.get("best_estimator")))
        self.log("AUC        : " + str(results.get("auc")))
        self.log("F1         : " + str(results.get("f1")))
        self.log("=" * 50)

        return state