"""
VEDA — Autonomous Data Science System
agents/deep_learning/dl_evaluation.py — DL Evaluation Agent

Evaluates all trained DL models:
- Confusion matrix
- ROC curve data
- Precision-Recall curve
- Model comparison table
- LLM-generated interpretation
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent

load_dotenv()


class DLEvaluationAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="DLEvaluationAgent",
            domain="deep_learning",
            version="1.0.0"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile"

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _load_dl_model(self, model_type: str, state: dict, input_size: int):
        """Load a saved DL model."""
        d = "outputs"
        suffix = "_" + model_type.lower() + "_model.pt"
        files = [f for f in os.listdir(d) if f.endswith(suffix)]
        if not files:
            return None

        model_path = os.path.join(d, sorted(files)[-1])

        try:
            if model_type == "mlp":
                from veda.agents.deep_learning.mlp import MLPNetwork
                hidden_sizes = [128, 64, 32] if input_size < 100 else [256, 128, 64]
                model = MLPNetwork(input_size, hidden_sizes, 1)
            elif model_type == "cnn":
                from veda.agents.deep_learning.cnn import CNNNetwork
                model = CNNNetwork(input_size, num_filters=32)
            elif model_type == "lstm":
                from veda.agents.deep_learning.lstm import LSTMNetwork
                model = LSTMNetwork(input_size, hidden_size=32)
            else:
                return None

            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.log("Failed to load " + model_type + " model: " + str(e), level="WARN")
            return None

    def _evaluate_model(self, model, X_val, y_val):
        """Evaluate a model and return metrics."""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            y_pred_proba = model(X_tensor).cpu().numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

        try:
            auc = round(float(roc_auc_score(y_val, y_pred_proba)), 4)
        except:
            auc = 0.5

        f1 = round(float(f1_score(y_val, y_pred, zero_division=0)), 4)
        acc = round(float(accuracy_score(y_val, y_pred)), 4)
        prec = round(float(precision_score(y_val, y_pred, zero_division=0)), 4)
        rec = round(float(recall_score(y_val, y_pred, zero_division=0)), 4)

        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        return {
            "auc": auc, "f1": f1, "accuracy": acc,
            "precision": prec, "recall": rec,
            "confusion_matrix": {
                "tp": int(tp), "fp": int(fp),
                "fn": int(fn), "tn": int(tn)
            }
        }

    def _generate_interpretation(self, comparison: dict, goal: str) -> str:
        """Generate LLM interpretation of DL results."""
        prompt = """You are VEDA. Interpret these deep learning model results.

Goal: """ + goal + """

Results:
""" + json.dumps(comparison, indent=2) + """

Write 3-4 sentences covering:
1. Which model performed best and why
2. What the metrics mean for the business goal
3. Any recommendations (use DL or stick with traditional ML?)

Be specific and concise."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "DL evaluation complete. See metrics for details."

    def run(self, state: dict) -> dict:
        """
        DL Evaluation:
        1. Load all trained DL models
        2. Evaluate each on held-out data
        3. Compare metrics
        4. Generate interpretation
        """

        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        goal = state.get("goal", "")

        self.log("Loading features for evaluation...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        # Prepare data
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        y = df[target_col].values
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
        y = y.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        input_size = X.shape[1]
        self.log("Evaluating DL models on " + str(len(X_val)) + " validation samples...")

        evaluation_results = {}

        for model_type in ["mlp", "cnn", "lstm"]:
            self.log("Evaluating " + model_type.upper() + "...")
            model = self._load_dl_model(model_type, state, input_size)
            if model:
                metrics = self._evaluate_model(model, X_val, y_val)
                evaluation_results[model_type.upper()] = metrics
                self.log(model_type.upper() + " AUC=" + str(metrics["auc"]) +
                        " F1=" + str(metrics["f1"]) +
                        " Acc=" + str(metrics["accuracy"]))
            else:
                self.log(model_type.upper() + " model not found — skipping", level="WARN")

        # Find best
        if evaluation_results:
            best = max(evaluation_results, key=lambda k: evaluation_results[k]["auc"])
            best_auc = evaluation_results[best]["auc"]
        else:
            best = "N/A"
            best_auc = 0.0

        # Generate interpretation
        self.log("Generating interpretation with Groq...")
        interpretation = self._generate_interpretation(evaluation_results, goal)

        # Save report
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        dl_eval = {
            "evaluation_results": evaluation_results,
            "best_model": best,
            "best_auc": best_auc,
            "interpretation": interpretation
        }

        report_path = "outputs/" + run_id + "_dl_evaluation.json"
        with open(report_path, "w") as f:
            json.dump(dl_eval, f, indent=2)

        state["dl_evaluation"] = dl_eval
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] DLEvaluationAgent: best=" +
            best + " AUC=" + str(best_auc)
        )

        self.log("=" * 50)
        self.log("DL EVALUATION COMPLETE")
        self.log("=" * 50)
        for model_name, metrics in evaluation_results.items():
            marker = " <- BEST" if model_name == best else ""
            self.log(model_name + " AUC=" + str(metrics["auc"]) +
                    " F1=" + str(metrics["f1"]) + marker)
        self.log("Interpretation: " + interpretation[:150] + "...")
        self.log("=" * 50)

        return state