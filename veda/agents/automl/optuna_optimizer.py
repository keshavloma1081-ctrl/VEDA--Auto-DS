"""
VEDA Optuna AutoML - Fix #7
Replaces basic grid search with Bayesian optimization.
"""
import os, logging, time
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class OptunaOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna.
    Finds better hyperparams in fewer trials than grid search.
    """

    def __init__(
        self,
        task_type: str = "classification",
        n_trials: int = 50,
        timeout: int = 300,
        metric: str = "roc_auc",
        direction: str = "maximize",
        n_jobs: int = 1
    ):
        self.task_type = task_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.direction = direction
        self.n_jobs = n_jobs
        self.best_params: Dict = {}
        self.best_score: float = 0.0
        self.study = None
        self.optimization_history: List[Dict] = []

    def optimize_xgboost(self, X_train, y_train, X_val, y_val) -> Dict:
        """Optimize XGBoost hyperparameters"""
        try:
            import optuna
            import xgboost as xgb
            from sklearn.metrics import roc_auc_score, r2_score

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "use_label_encoder": False,
                    "eval_metric": "logloss" if self.task_type == "classification" else "rmse",
                    "verbosity": 0,
                    "random_state": 42
                }
                if self.task_type == "classification":
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              early_stopping_rounds=50,
                              verbose=False)
                    preds = model.predict_proba(X_val)[:, 1]
                    return roc_auc_score(y_val, preds)
                else:
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              early_stopping_rounds=50,
                              verbose=False)
                    preds = model.predict(X_val)
                    return r2_score(y_val, preds)

            self.study = optuna.create_study(direction=self.direction)
            self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            log.info(f"XGBoost optimized | trials={len(self.study.trials)} | best={self.best_score:.4f}")
            return self.best_params

        except ImportError as e:
            log.warning(f"Optuna/XGBoost not available: {e}. Using defaults.")
            return self._xgboost_defaults()

    def optimize_lightgbm(self, X_train, y_train, X_val, y_val) -> Dict:
        """Optimize LightGBM hyperparameters"""
        try:
            import optuna
            import lightgbm as lgb
            from sklearn.metrics import roc_auc_score, r2_score

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "random_state": 42,
                    "verbosity": -1,
                    "force_col_wise": True
                }
                if self.task_type == "classification":
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                    preds = model.predict_proba(X_val)[:, 1]
                    return roc_auc_score(y_val, preds)
                else:
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                    preds = model.predict(X_val)
                    return r2_score(y_val, preds)

            self.study = optuna.create_study(direction=self.direction)
            self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            log.info(f"LightGBM optimized | trials={len(self.study.trials)} | best={self.best_score:.4f}")
            return self.best_params

        except ImportError as e:
            log.warning(f"Optuna/LightGBM not available: {e}. Using defaults.")
            return self._lightgbm_defaults()

    def optimize_random_forest(self, X_train, y_train, X_val, y_val) -> Dict:
        """Optimize Random Forest hyperparameters"""
        try:
            import optuna
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import roc_auc_score, r2_score

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "random_state": 42,
                    "n_jobs": -1
                }
                if self.task_type == "classification":
                    model = RandomForestClassifier(**params)
                    model.fit(X_train, y_train)
                    preds = model.predict_proba(X_val)[:, 1]
                    return roc_auc_score(y_val, preds)
                else:
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    return r2_score(y_val, preds)

            self.study = optuna.create_study(direction=self.direction)
            self.study.optimize(objective, n_trials=min(self.n_trials, 30), timeout=self.timeout)
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            log.info(f"RandomForest optimized | trials={len(self.study.trials)} | best={self.best_score:.4f}")
            return self.best_params

        except ImportError as e:
            log.warning(f"Optuna not available: {e}. Using defaults.")
            return self._rf_defaults()

    def optimize_all(self, X_train, y_train, X_val, y_val) -> Dict:
        """
        Run Optuna on all models and return the best one.
        This is the main AutoML entry point.
        """
        results = {}
        start_time = time.time()

        models_to_try = [
            ("xgboost", self.optimize_xgboost),
            ("lightgbm", self.optimize_lightgbm),
            ("random_forest", self.optimize_random_forest),
        ]

        for model_name, optimizer_func in models_to_try:
            try:
                log.info(f"Optimizing {model_name}...")
                t0 = time.time()
                best_params = optimizer_func(X_train, y_train, X_val, y_val)
                duration = round(time.time() - t0, 2)

                results[model_name] = {
                    "best_params": best_params,
                    "best_score": self.best_score,
                    "optimization_time_seconds": duration
                }

                self.optimization_history.append({
                    "model": model_name,
                    "score": self.best_score,
                    "params": best_params,
                    "duration": duration
                })

                log.info(f"{model_name}: score={self.best_score:.4f} in {duration}s")

            except Exception as e:
                log.error(f"Failed to optimize {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        # Find best model
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if not valid_results:
            return {"error": "All optimizations failed", "results": results}

        best_model = max(valid_results.items(), key=lambda x: x[1]["best_score"])
        total_time = round(time.time() - start_time, 2)

        return {
            "best_model": best_model[0],
            "best_score": best_model[1]["best_score"],
            "best_params": best_model[1]["best_params"],
            "all_results": results,
            "total_optimization_time": total_time,
            "metric": self.metric,
        }

    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization run"""
        if not self.study:
            return {"message": "No optimization run yet"}
        return {
            "n_trials": len(self.study.trials),
            "best_score": round(self.best_score, 4),
            "best_params": self.best_params,
            "metric": self.metric,
            "direction": self.direction,
            "optimization_history": self.optimization_history
        }

    # ── Defaults (when Optuna not available) ──────────────────────────────

    def _xgboost_defaults(self) -> Dict:
        if self.task_type == "classification":
            return {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                    "subsample": 0.8, "colsample_bytree": 0.8, "use_label_encoder": False}
        return {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8}

    def _lightgbm_defaults(self) -> Dict:
        return {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "num_leaves": 50, "subsample": 0.8, "verbosity": -1}

    def _rf_defaults(self) -> Dict:
        return {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5,
                "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42}


# ─────────────────────────────────────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(model, X, y, cv=5, task_type="classification") -> Dict:
    """Proper k-fold cross validation"""
    try:
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
        from sklearn.metrics import make_scorer, roc_auc_score

        if task_type == "classification":
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scorer = make_scorer(roc_auc_score, needs_proba=True)
            metric = "roc_auc"
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            scorer = "r2"
            metric = "r2"

        scores = cross_val_score(model, X, y, cv=kf, scoring=scorer, n_jobs=-1)

        return {
            "metric": metric,
            "cv_folds": cv,
            "mean_score": round(float(np.mean(scores)), 4),
            "std_score": round(float(np.std(scores)), 4),
            "min_score": round(float(np.min(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
            "fold_scores": [round(float(s), 4) for s in scores]
        }

    except Exception as e:
        log.error(f"Cross validation failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: List[str]) -> Dict:
    """Extract feature importance from trained model"""
    try:
        importance_dict = {}

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_dict = {
                feat: round(float(imp), 4)
                for feat, imp in zip(feature_names, importances)
            }
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_).flatten()
            importance_dict = {
                feat: round(float(coef), 4)
                for feat, coef in zip(feature_names, coefs)
            }

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        # Top 10 features
        top_features = list(importance_dict.items())[:10]

        return {
            "feature_importance": importance_dict,
            "top_10_features": [
                {"feature": f, "importance": imp}
                for f, imp in top_features
            ],
            "most_important": top_features[0][0] if top_features else None
        }

    except Exception as e:
        log.error(f"Feature importance extraction failed: {e}")
        return {"error": str(e)}
