"""
VEDA — Autonomous Data Science System
agents/automl/model_compression.py — Model Compression Agent

Model compression techniques:
- Feature reduction
- Model pruning simulation
- Quantization simulation
- Distillation simulation
- Size vs performance tradeoff
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from veda.core.base_agent import BaseAgent


class ModelCompressionAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="ModelCompressionAgent",
            domain="automl",
            version="1.0.0"
        )

    def _load_model(self, state):
        model_info = state.get("model_info", {})
        model_path = model_info.get("model_path")
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_model.pkl")]
        if not files:
            return None
        return joblib.load(os.path.join(d, sorted(files)[-1]))

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _encode_df(self, df):
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=["object", "bool"]).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        return df_enc

    def _estimate_model_size(self, model) -> dict:
        """Estimate model size in memory."""
        import pickle
        try:
            model_bytes = len(pickle.dumps(model))
            return {
                "size_bytes": model_bytes,
                "size_kb": round(model_bytes / 1024, 2),
                "size_mb": round(model_bytes / (1024 * 1024), 4)
            }
        except:
            return {"size_bytes": 0, "size_kb": 0, "size_mb": 0}

    def _feature_pruning(self, model, X_train, X_test, y_test,
                          feature_names: list, top_k: int = 10) -> dict:
        """Compress model by using only top features."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[::-1][:top_k]

                X_test_pruned = X_test[:, top_indices]
                X_train_pruned = X_train[:, top_indices]

                import lightgbm as lgb
                pruned_model = lgb.LGBMClassifier(
                    n_estimators=50, random_state=42, verbose=-1
                )
                pruned_model.fit(X_train_pruned, y_test[:len(X_train_pruned)]
                                if len(y_test) >= len(X_train_pruned) else y_test)
                y_pred = pruned_model.predict_proba(X_test_pruned)[:, 1]
                auc = round(float(roc_auc_score(y_test, y_pred)), 4)

                pruned_size = self._estimate_model_size(pruned_model)
                return {
                    "features_kept": top_k,
                    "original_features": len(feature_names),
                    "pruned_auc": auc,
                    "pruned_size_kb": pruned_size["size_kb"]
                }
        except Exception as e:
            self.log("Feature pruning failed: " + str(e), level="WARN")
        return {}

    def _simulate_quantization(self, model_size_kb: float) -> dict:
        """Simulate INT8 quantization effects."""
        quantized_size = model_size_kb * 0.25
        estimated_auc_drop = 0.005
        return {
            "original_size_kb": model_size_kb,
            "quantized_size_kb": round(quantized_size, 2),
            "compression_ratio": 4.0,
            "estimated_auc_drop": estimated_auc_drop,
            "technique": "INT8 Quantization (simulated)"
        }

    def _simulate_distillation(self, teacher_auc: float,
                                model_size_kb: float) -> dict:
        """Simulate knowledge distillation."""
        student_size = model_size_kb * 0.3
        student_auc = teacher_auc - 0.02
        return {
            "teacher_auc": teacher_auc,
            "student_auc": round(student_auc, 4),
            "teacher_size_kb": model_size_kb,
            "student_size_kb": round(student_size, 2),
            "compression_ratio": round(model_size_kb / max(student_size, 1), 2),
            "technique": "Knowledge Distillation (simulated)"
        }

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading model and features...")
        model = self._load_model(state)
        df = self._load_features(state)

        if model is None or df is None:
            self.log("Model or features not found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        df_enc = self._encode_df(df)
        feature_cols = [c for c in df.columns if c != target_col]
        X = df_enc[feature_cols].fillna(0).values.astype(np.float32)
        y_raw = df[target_col].values
        if y_raw.dtype == object:
            y_raw = LabelEncoder().fit_transform(y_raw.astype(str))
        y = y_raw.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Baseline model size
        self.log("Estimating model size...")
        model_size = self._estimate_model_size(model)
        self.log("Model size: " + str(model_size["size_kb"]) + " KB")

        # Baseline AUC
        baseline_auc = state.get("model_info", {}).get(
            "test_metrics", {}
        ).get("auc_roc", 0.8)

        # Feature pruning
        self.log("Running feature pruning...")
        pruning_results = self._feature_pruning(
            model, X_train, X_test, y_test, feature_cols, top_k=10
        )

        # Quantization simulation
        self.log("Simulating quantization...")
        quant_results = self._simulate_quantization(model_size["size_kb"])

        # Distillation simulation
        self.log("Simulating knowledge distillation...")
        distill_results = self._simulate_distillation(
            baseline_auc, model_size["size_kb"]
        )

        compression_results = {
            "original_model_size": model_size,
            "baseline_auc": baseline_auc,
            "feature_pruning": pruning_results,
            "quantization": quant_results,
            "distillation": distill_results,
            "recommendations": [
                "Use INT8 quantization for 4x size reduction with minimal AUC loss",
                "Apply feature pruning to reduce inference latency",
                "Consider knowledge distillation for edge deployment"
            ]
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_compression_results.json"
        with open(path, "w") as f:
            json.dump(compression_results, f, indent=2)

        state["compression_results"] = compression_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ModelCompressionAgent: " +
            "size=" + str(model_size["size_kb"]) + "KB " +
            "quant_ratio=4x"
        )

        self.log("=" * 50)
        self.log("MODEL COMPRESSION COMPLETE")
        self.log("Original size : " + str(model_size["size_kb"]) + " KB")
        self.log("After quant   : " + str(quant_results["quantized_size_kb"]) + " KB")
        self.log("After distill : " + str(distill_results["student_size_kb"]) + " KB")
        self.log("=" * 50)

        return state