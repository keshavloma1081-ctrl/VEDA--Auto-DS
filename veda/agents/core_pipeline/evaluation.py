import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from veda.core.base_agent import BaseAgent

class EvaluationAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="EvaluationAgent", domain="ml", version="1.0.0")

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _load_model(self, state):
        model_info = state.get("model_info", {})
        model_path = model_info.get("model_path")
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_model.pkl")]
        if not files:
            raise FileNotFoundError("No model found in outputs/")
        return joblib.load(os.path.join(d, sorted(files)[-1]))

    def run(self, state):
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading feature matrix and model...")
        df = self._load_features(state)
        model = self._load_model(state)

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]
            self.log("Using last column as target: " + str(target_col), level="WARN")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.log("Running evaluation on " + str(len(y)) + " samples...")

        # Predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Metrics
        auc = round(float(roc_auc_score(y, y_proba)), 4)
        f1 = round(float(f1_score(y, y_pred)), 4)
        acc = round(float(accuracy_score(y, y_pred)), 4)
        prec = round(float(precision_score(y, y_pred)), 4)
        rec = round(float(recall_score(y, y_pred)), 4)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Classification report
        report = classification_report(y, y_pred)

        # Pass/fail threshold
        threshold = 0.70
        passed = auc >= threshold

        # Save evaluation report
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        eval_path = "outputs/" + run_id + "_evaluation.json"

        eval_results = {
            "auc_roc": auc,
            "f1_score": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)
            },
            "passed_threshold": passed,
            "threshold": threshold,
            "classification_report": report
        }

        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        # Update state
        state.setdefault("model_info", {})
        state["model_info"]["passed_evaluation"] = passed
        state["model_info"]["test_metrics"] = {
            "auc_roc": auc,
            "f1_score": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec
        }

        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] EvaluationAgent: AUC=" + str(auc) +
            " F1=" + str(f1) + " PASS=" + str(passed)
        )

        # Print summary
        self.log("=" * 50)
        self.log("EVALUATION RESULTS")
        self.log("AUC-ROC   : " + str(auc) + (" PASS" if passed else " FAIL"))
        self.log("F1 Score  : " + str(f1))
        self.log("Accuracy  : " + str(acc))
        self.log("Precision : " + str(prec))
        self.log("Recall    : " + str(rec))
        self.log("Confusion Matrix:")
        self.log("  TP=" + str(tp) + " FP=" + str(fp))
        self.log("  FN=" + str(fn) + " TN=" + str(tn))
        self.log("Threshold : " + str(threshold))
        self.log("Result    : " + ("PASSED" if passed else "FAILED"))
        self.log("=" * 50)

        return state