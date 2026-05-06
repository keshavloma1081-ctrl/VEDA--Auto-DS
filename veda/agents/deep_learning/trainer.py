"""
VEDA — Autonomous Data Science System
agents/deep_learning/trainer.py — DL Trainer Agent

Compares MLP, CNN, LSTM and selects the best.
Runs all three and picks winner by validation AUC.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent


class DLTrainerAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="DLTrainerAgent",
            domain="deep_learning",
            version="1.0.0"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def run(self, state: dict) -> dict:
        """
        DL Trainer:
        1. Run MLP, CNN, LSTM
        2. Compare all three
        3. Select best model
        4. Save comparison report
        """

        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        self.log("Running all 3 DL models for comparison...")
        self.log("Input shape: " + str(df.shape))

        results = {}

        # Run MLP
        self.log("\n--- Training MLP ---")
        try:
            from veda.agents.deep_learning.mlp import MLPAgent
            mlp = MLPAgent()
            state = mlp.execute(state)
            mlp_res = state.get("dl_results", {})
            if isinstance(mlp_res, dict) and "auc" in mlp_res:
                results["MLP"] = {"auc": mlp_res["auc"], "f1": mlp_res["f1"],
                                   "params": mlp_res["total_params"]}
        except Exception as e:
            self.log("MLP failed: " + str(e), level="WARN")

        # Run CNN
        self.log("\n--- Training CNN ---")
        try:
            from veda.agents.deep_learning.cnn import CNNAgent
            cnn = CNNAgent()
            state = cnn.execute(state)
            cnn_res = state.get("dl_results", {}).get("cnn", {})
            if cnn_res:
                results["CNN"] = {"auc": cnn_res["auc"], "f1": cnn_res["f1"],
                                   "params": cnn_res["total_params"]}
        except Exception as e:
            self.log("CNN failed: " + str(e), level="WARN")

        # Run LSTM
        self.log("\n--- Training LSTM ---")
        try:
            from veda.agents.deep_learning.lstm import LSTMAgent
            lstm = LSTMAgent()
            state = lstm.execute(state)
            lstm_res = state.get("dl_results", {}).get("lstm", {})
            if lstm_res:
                results["LSTM"] = {"auc": lstm_res["auc"], "f1": lstm_res["f1"],
                                    "params": lstm_res["total_params"]}
        except Exception as e:
            self.log("LSTM failed: " + str(e), level="WARN")

        # Select best
        if results:
            best_model = max(results, key=lambda k: results[k]["auc"])
            best_auc = results[best_model]["auc"]
        else:
            best_model = "MLP"
            best_auc = 0.0

        # Print comparison
        self.log("=" * 50)
        self.log("DL MODEL COMPARISON")
        self.log("=" * 50)
        for model_name, metrics in results.items():
            marker = " <- WINNER" if model_name == best_model else ""
            self.log(model_name.ljust(6) + " AUC=" + str(metrics["auc"]) +
                    " F1=" + str(metrics["f1"]) +
                    " params=" + str(metrics["params"]) + marker)
        self.log("=" * 50)
        self.log("Best DL model: " + best_model + " (AUC=" + str(best_auc) + ")")

        # Save comparison report
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        comparison = {
            "results": results,
            "winner": best_model,
            "best_auc": best_auc,
            "device": str(self.device)
        }

        report_path = "outputs/" + run_id + "_dl_comparison.json"
        with open(report_path, "w") as f:
            json.dump(comparison, f, indent=2)

        state["dl_comparison"] = comparison
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] DLTrainerAgent: winner=" +
            best_model + " AUC=" + str(best_auc)
        )

        return state