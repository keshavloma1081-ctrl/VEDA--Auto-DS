"""
VEDA — Autonomous Data Science System
agents/mlops/retraining.py — Model Retraining Agent

Auto-retrains model when drift is detected:
- Checks drift report
- Triggers retraining pipeline
- Compares new vs old model
- Promotes if improved
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent


class RetrainingAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="RetrainingAgent",
            domain="mlops",
            version="1.0.0"
        )

    def _load_features(self, state: dict) -> pd.DataFrame:
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _load_current_model(self, state: dict):
        model_info = state.get("model_info", {})
        model_path = model_info.get("model_path")
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_model.pkl")]
        if not files:
            return None
        return joblib.load(os.path.join(d, sorted(files)[-1]))

    def _retrain_model(self, X, y, model_name: str):
        """Retrain model with updated data."""
        models = {
            "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42)
        }

        model = models.get(model_name, models["LightGBM"])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        model.fit(X, y)
        return model, round(float(scores.mean()), 4), round(float(scores.std()), 4)

    def run(self, state: dict) -> dict:
        """
        Retraining Pipeline:
        1. Check if retraining needed
        2. Load latest data
        3. Retrain model
        4. Compare with current
        5. Promote if better
        """

        drift_report = state.get("drift_report", {})
        needs_retraining = drift_report.get("needs_retraining", False)

        if not needs_retraining:
            self.log("No retraining needed — drift score below threshold")
            state.setdefault("planner_decision_log", []).append(
                "[" + datetime.now().isoformat() + "] RetrainingAgent: skipped — no drift detected"
            )
            return state

        self.log("Drift detected — triggering retraining...")

        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        model_info = state.get("model_info", {})
        model_name = model_info.get("model_name", "LightGBM")
        current_auc = model_info.get("test_metrics", {}).get("auc_roc", 0)

        self.log("Loading training data...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found for retraining", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        X = df.drop(columns=[target_col]).values.astype(np.float32)
        y = df[target_col].values
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
        X = np.nan_to_num(X, nan=0.0)

        self.log("Retraining " + model_name + " on " + str(X.shape[0]) + " samples...")
        new_model, new_auc, new_std = self._retrain_model(X, y, model_name)

        self.log("Current AUC : " + str(current_auc))
        self.log("New AUC     : " + str(new_auc) + " +/- " + str(new_std))

        # Promote if improved
        improved = new_auc > current_auc
        if improved:
            self.log("New model is BETTER — promoting to production")
            os.makedirs("outputs", exist_ok=True)
            run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
            new_model_path = "outputs/" + run_id + "_retrained_model.pkl"
            joblib.dump(new_model, new_model_path)

            state.setdefault("model_info", {})
            state["model_info"]["model_path"] = new_model_path
            state["model_info"]["test_metrics"] = {"auc_roc": new_auc}
            self.log("New model saved to: " + new_model_path)
        else:
            self.log("New model is NOT better — keeping current model")

        retraining_results = {
            "triggered": True,
            "drift_score": drift_report.get("drift_score", 0),
            "current_auc": current_auc,
            "new_auc": new_auc,
            "new_auc_std": new_std,
            "improved": improved,
            "promoted": improved,
            "model_name": model_name
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        results_path = "outputs/" + run_id + "_retraining_results.json"
        with open(results_path, "w") as f:
            json.dump(retraining_results, f, indent=2)

        state["retraining_results"] = retraining_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] RetrainingAgent: new_auc=" +
            str(new_auc) + " improved=" + str(improved) + " promoted=" + str(improved)
        )

        self.log("=" * 50)
        self.log("RETRAINING COMPLETE")
        self.log("Current AUC : " + str(current_auc))
        self.log("New AUC     : " + str(new_auc))
        self.log("Improved    : " + str(improved))
        self.log("Promoted    : " + str(improved))
        self.log("=" * 50)

        return state