"""
VEDA — Autonomous Data Science System
agents/streaming/online_learning.py — Online Learning Agent

Incremental/online learning using River:
- Hoeffding Tree classifier
- Adaptive Random Forest
- Logistic regression online
- Concept drift detection
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from veda.core.base_agent import BaseAgent


class OnlineLearningAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="OnlineLearningAgent",
            domain="streaming",
            version="1.0.0"
        )

    def _load_stream_features(self, state):
        stream_features = state.get("stream_features")
        if stream_features is not None:
            return stream_features if isinstance(stream_features, pd.DataFrame) \
                   else pd.DataFrame(stream_features)
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_stream_features.parquet")]
        if files:
            return pd.read_parquet(os.path.join(d, sorted(files)[-1]))
        files = [f for f in os.listdir(d) if f.endswith("_stream_data.parquet")]
        if files:
            return pd.read_parquet(os.path.join(d, sorted(files)[-1]))
        return None

    def _river_hoeffding_tree(self, X: list, y: list) -> dict:
        """Train Hoeffding Tree online."""
        try:
            from river import tree, metrics
            model = tree.HoeffdingTreeClassifier()
            metric = metrics.Accuracy()
            accuracies = []

            for i, (xi, yi) in enumerate(zip(X, y)):
                pred = model.predict_one(xi)
                if pred is not None:
                    metric.update(yi, pred)
                model.learn_one(xi, yi)
                if i % 100 == 0 and i > 0:
                    accuracies.append(round(float(metric.get()), 4))

            return {
                "model": "HoeffdingTree",
                "final_accuracy": round(float(metric.get()), 4),
                "accuracy_history": accuracies,
                "n_samples": len(X)
            }
        except Exception as e:
            self.log("Hoeffding Tree failed: " + str(e), level="WARN")
            return self._sklearn_online_fallback(X, y)

    def _river_adaptive_random_forest(self, X: list, y: list) -> dict:
        """Train Adaptive Random Forest online."""
        try:
            from river import ensemble, metrics
            model = ensemble.AdaptiveRandomForestClassifier(
                n_models=5, seed=42
            )
            metric = metrics.Accuracy()
            accuracies = []

            for i, (xi, yi) in enumerate(zip(X, y)):
                pred = model.predict_one(xi)
                if pred is not None:
                    metric.update(yi, pred)
                model.learn_one(xi, yi)
                if i % 100 == 0 and i > 0:
                    accuracies.append(round(float(metric.get()), 4))

            return {
                "model": "AdaptiveRandomForest",
                "final_accuracy": round(float(metric.get()), 4),
                "accuracy_history": accuracies,
                "n_samples": len(X)
            }
        except Exception as e:
            self.log("ARF failed: " + str(e), level="WARN")
            return {"model": "ARF", "error": str(e)}

    def _sklearn_online_fallback(self, X: list, y: list) -> dict:
        """SGD-based online learning fallback."""
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        X_arr = np.array([list(xi.values()) for xi in X])
        y_arr = np.array(y)

        scaler = StandardScaler()
        model = SGDClassifier(random_state=42, max_iter=1)

        batch_size = 50
        accuracies = []

        for i in range(0, len(X_arr), batch_size):
            X_batch = X_arr[i:i+batch_size]
            y_batch = y_arr[i:i+batch_size]
            X_scaled = scaler.partial_fit(X_batch).transform(X_batch)
            model.partial_fit(X_scaled, y_batch, classes=np.unique(y_arr))
            if i > 0:
                preds = model.predict(X_scaled)
                acc = round(float((preds == y_batch).mean()), 4)
                accuracies.append(acc)

        final_acc = accuracies[-1] if accuracies else 0.5
        return {
            "model": "SGDClassifier",
            "final_accuracy": final_acc,
            "accuracy_history": accuracies,
            "n_samples": len(X_arr)
        }

    def _detect_concept_drift(self, accuracy_history: list) -> dict:
        """Detect concept drift from accuracy trend."""
        if len(accuracy_history) < 3:
            return {"drift_detected": False}

        recent = accuracy_history[-3:]
        earlier = accuracy_history[:3]
        drop = float(np.mean(earlier) - np.mean(recent))

        return {
            "drift_detected": bool(drop > 0.05),
            "accuracy_drop": round(drop, 4),
            "early_accuracy": round(float(np.mean(earlier)), 4),
            "recent_accuracy": round(float(np.mean(recent)), 4)
        }

    def run(self, state: dict) -> dict:
        self.log("Loading stream features...")
        df = self._load_stream_features(state)

        if df is None:
            self.log("No stream data found", level="WARN")
            return state

        self.log("Stream data: " + str(df.shape))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["is_fraud", "hour", "minute"]][:8]

        if "is_fraud" in df.columns:
            target_col = "is_fraud"
        else:
            target_col = numeric_cols[-1] if numeric_cols else None

        if not target_col or target_col not in df.columns:
            self.log("No target column found", level="WARN")
            return state

        df_clean = df[feature_cols + [target_col]].fillna(0)
        X = [dict(row) for _, row in df_clean[feature_cols].iterrows()]
        y = (df_clean[target_col] > df_clean[target_col].median()).astype(int).tolist()

        self.log("Training Hoeffding Tree online...")
        ht_results = self._river_hoeffding_tree(X, y)
        self.log("HT accuracy: " + str(ht_results.get("final_accuracy")))

        self.log("Training Adaptive Random Forest online...")
        arf_results = self._river_adaptive_random_forest(X, y)
        self.log("ARF accuracy: " + str(arf_results.get("final_accuracy")))

        self.log("Detecting concept drift...")
        drift = self._detect_concept_drift(
            ht_results.get("accuracy_history", [])
        )
        self.log("Drift detected: " + str(drift["drift_detected"]))

        best_model = "HoeffdingTree"
        if arf_results.get("final_accuracy", 0) > ht_results.get("final_accuracy", 0):
            best_model = "AdaptiveRandomForest"

        online_results = {
            "hoeffding_tree": ht_results,
            "adaptive_random_forest": arf_results,
            "concept_drift": drift,
            "best_model": best_model,
            "n_samples_trained": len(X)
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_online_learning.json"
        with open(path, "w") as f:
            json.dump(online_results, f, indent=2)

        state["online_learning"] = online_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] OnlineLearningAgent: " +
            "best=" + best_model +
            " HT_acc=" + str(ht_results.get("final_accuracy"))
        )

        self.log("=" * 50)
        self.log("ONLINE LEARNING COMPLETE")
        self.log("Best model  : " + best_model)
        self.log("HT accuracy : " + str(ht_results.get("final_accuracy")))
        self.log("ARF accuracy: " + str(arf_results.get("final_accuracy")))
        self.log("Drift       : " + str(drift["drift_detected"]))
        self.log("=" * 50)

        return state