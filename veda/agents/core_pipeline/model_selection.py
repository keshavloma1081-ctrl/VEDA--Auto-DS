import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import lightgbm as lgb
from veda.core.base_agent import BaseAgent

class ModelSelectionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ModelSelectionAgent", domain="ml", version="1.0.0")

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _get_candidates(self):
        return {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric="logloss", verbosity=0),
            "LightGBM": lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1),
            "Baseline": DummyClassifier(strategy="most_frequent")
        }

    def run(self, state):
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading feature matrix...")
        df = self._load_features(state)

        if not target_col or target_col not in df.columns:
            self.log("No target column found — using last column as target", level="WARN")
            target_col = df.columns[-1]

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.log("Shape: " + str(X.shape) + " features, target: " + str(target_col))
        self.log("Benchmarking 5 model candidates...")

        candidates = self._get_candidates()
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results = []

        for name, model in candidates.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
                mean_auc = round(float(scores.mean()), 4)
                std_auc = round(float(scores.std()), 4)
                results.append({"model": name, "auc": mean_auc, "std": std_auc})
                self.log(name + " AUC: " + str(mean_auc) + " +/- " + str(std_auc))
            except Exception as e:
                self.log("FAILED " + name + ": " + str(e), level="WARN")
                results.append({"model": name, "auc": 0.0, "std": 0.0})

        results_sorted = sorted(results, key=lambda x: x["auc"], reverse=True)
        best = results_sorted[0]

        self.log("=" * 50)
        self.log("BENCHMARK RESULTS:")
        for r in results_sorted:
            self.log("  " + r["model"] + ": " + str(r["auc"]) + " +/- " + str(r["std"]))
        self.log("WINNER: " + best["model"] + " (AUC=" + str(best["auc"]) + ")")
        self.log("=" * 50)

        state["benchmark_table"] = json.dumps(results_sorted)
        state.setdefault("model_info", {})
        state["model_info"]["model_name"] = best["model"]
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ModelSelectionAgent: winner=" + best["model"] + " AUC=" + str(best["auc"])
        )

        return state