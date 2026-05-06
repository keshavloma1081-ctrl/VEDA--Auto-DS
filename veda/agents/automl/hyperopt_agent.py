"""
VEDA — Autonomous Data Science System
agents/automl/hyperopt_agent.py — Hyperparameter Optimization Agent

Hyperparameter optimization using Optuna:
- LightGBM hyperparameter tuning
- Bayesian optimization
- Trial pruning
- Best params reporting
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent


class HyperoptAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="HyperoptAgent",
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

    def _optuna_optimize(self, X, y, n_trials: int = 20) -> dict:
        """Optimize LightGBM with Optuna."""
        try:
            import optuna
            import lightgbm as lgb
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "random_state": 42,
                    "verbose": -1
                }
                model = lgb.LGBMClassifier(**params)
                scores = cross_val_score(model, X, y, cv=cv,
                                        scoring="roc_auc", n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best_params = study.best_params
            best_value = study.best_value

            self.log("Best AUC: " + str(round(best_value, 4)))
            self.log("Best params: " + str(best_params))

            return {
                "method": "Optuna Bayesian Optimization",
                "n_trials": n_trials,
                "best_params": best_params,
                "best_auc": round(float(best_value), 4),
                "n_trials_completed": len(study.trials),
                "status": "success"
            }

        except Exception as e:
            self.log("Optuna failed: " + str(e), level="WARN")
            return self._grid_search_fallback(X, y)

    def _grid_search_fallback(self, X, y) -> dict:
        """Grid search fallback."""
        import lightgbm as lgb
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        param_grid = [
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
            {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08},
        ]

        best_params = None
        best_auc = 0

        for params in param_grid:
            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            if scores.mean() > best_auc:
                best_auc = scores.mean()
                best_params = params

        return {
            "method": "Grid Search Fallback",
            "best_params": best_params,
            "best_auc": round(float(best_auc), 4),
            "status": "success"
        }

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features for hyperparameter optimization...")
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

        self.log("Running Optuna optimization (20 trials)...")
        results = self._optuna_optimize(X, y, n_trials=20)

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_hyperopt_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        state["hyperopt_results"] = results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] HyperoptAgent: " +
            "best_auc=" + str(results.get("best_auc")) +
            " method=" + str(results.get("method"))
        )

        self.log("=" * 50)
        self.log("HYPEROPT COMPLETE")
        self.log("Method    : " + str(results.get("method")))
        self.log("Best AUC  : " + str(results.get("best_auc")))
        self.log("Best params: " + str(results.get("best_params")))
        self.log("=" * 50)

        return state