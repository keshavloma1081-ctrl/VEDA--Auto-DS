import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from veda.core.base_agent import BaseAgent

class TrainingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="TrainingAgent", domain="ml", version="1.0.0")

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _get_model(self, model_name):
        models = {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
            "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            "Baseline": DummyClassifier(strategy="most_frequent")
        }
        return models.get(model_name, lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))

    def run(self, state):
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        model_info = state.get("model_info", {})
        model_name = model_info.get("model_name", "LightGBM")

        self.log("Loading feature matrix...")
        df = self._load_features(state)

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]
            self.log("Using last column as target: " + str(target_col), level="WARN")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.log("Training " + model_name + " on " + str(X.shape[0]) + " rows x " + str(X.shape[1]) + " features")

        model = self._get_model(model_name)

        # Cross validation
        self.log("Running 5-fold cross validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        cv_mean = round(float(cv_scores.mean()), 4)
        cv_std = round(float(cv_scores.std()), 4)
        self.log("CV AUC: " + str(cv_mean) + " +/- " + str(cv_std))

        # Train final model on full data
        self.log("Training final model on full dataset...")
        model.fit(X, y)

        # Evaluate on training set
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        train_auc = round(float(roc_auc_score(y, y_proba)), 4)
        train_f1 = round(float(f1_score(y, y_pred)), 4)
        train_acc = round(float(accuracy_score(y, y_pred)), 4)

        self.log("Train AUC: " + str(train_auc))
        self.log("Train F1:  " + str(train_f1))
        self.log("Train Acc: " + str(train_acc))

        # Save model
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        model_path = "outputs/" + run_id + "_model.pkl"
        joblib.dump(model, model_path)
        self.log("Model saved to: " + model_path)

        # Save feature names
        feature_names = list(X.columns)
        features_path = "outputs/" + run_id + "_feature_names.json"
        with open(features_path, "w") as f:
            json.dump(feature_names, f)

        # MLflow logging
        try:
            mlflow.set_experiment("VEDA")
            with mlflow.start_run(run_name=model_name + "_" + run_id):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("n_rows", X.shape[0])
                mlflow.log_metric("cv_auc_mean", cv_mean)
                mlflow.log_metric("cv_auc_std", cv_std)
                mlflow.log_metric("train_auc", train_auc)
                mlflow.log_metric("train_f1", train_f1)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.sklearn.log_model(model, "model")
                mlflow_run_id = mlflow.active_run().info.run_id
            self.log("MLflow run logged: " + mlflow_run_id)
        except Exception as e:
            self.log("MLflow logging skipped: " + str(e), level="WARN")
            mlflow_run_id = None

        # Update state
        state.setdefault("model_info", {})
        state["model_info"]["model_name"] = model_name
        state["model_info"]["model_path"] = model_path
        state["model_info"]["mlflow_run_id"] = mlflow_run_id
        state["model_info"]["cv_metrics"] = {
            "auc_roc": cv_mean,
            "auc_std": cv_std
        }
        state["model_info"]["test_metrics"] = {
            "auc_roc": train_auc,
            "f1_score": train_f1,
            "accuracy": train_acc
        }

        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] TrainingAgent: " + model_name + " trained CV_AUC=" + str(cv_mean)
        )

        self.log("=" * 50)
        self.log("TRAINING COMPLETE")
        self.log("Model    : " + model_name)
        self.log("CV AUC   : " + str(cv_mean) + " +/- " + str(cv_std))
        self.log("Train AUC: " + str(train_auc))
        self.log("Saved to : " + model_path)
        self.log("=" * 50)

        return state