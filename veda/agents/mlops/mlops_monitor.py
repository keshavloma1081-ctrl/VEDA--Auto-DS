"""
VEDA — Autonomous Data Science System
agents/mlops/mlops_monitor.py — MLOps Monitor Agent

Monitors model in production:
- Prediction latency tracking
- Throughput metrics
- Error rate monitoring
- Prometheus metrics export
- Alert generation
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from veda.core.base_agent import BaseAgent


class MLOpsMonitorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="MLOpsMonitorAgent",
            domain="mlops",
            version="1.0.0"
        )

    def _load_model(self, state: dict):
        model_info = state.get("model_info", {})
        model_path = model_info.get("model_path")
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_model.pkl")]
        if not files:
            return None
        return joblib.load(os.path.join(d, sorted(files)[-1]))

    def _load_features(self, state: dict) -> pd.DataFrame:
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _simulate_prediction_requests(self, model, X: np.ndarray,
                                       n_requests: int = 100) -> dict:
        """Simulate production prediction requests and track metrics."""
        latencies = []
        errors = 0
        predictions = []

        sample_size = min(n_requests, len(X))
        indices = np.random.choice(len(X), sample_size, replace=True)

        for idx in indices:
            start = time.perf_counter()
            try:
                sample = X[idx:idx+1]
                pred = model.predict(sample)[0]
                proba = model.predict_proba(sample)[0][1]
                predictions.append({"pred": int(pred), "proba": float(proba)})
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
            except Exception:
                errors += 1

        if not latencies:
            latencies = [0.0]
        latencies = np.array(latencies)
        return {
        
            "total_requests": n_requests,
            "successful": len(latencies),
            "errors": errors,
            "error_rate": round(errors / n_requests, 4),
            "latency_mean_ms": round(float(np.mean(latencies)), 4),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 4),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 4),
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 4),
            "latency_max_ms": round(float(np.max(latencies)), 4),
            "positive_rate": round(sum(p["pred"] for p in predictions) / max(len(predictions), 1), 4),
            "avg_confidence": round(float(np.mean([p["proba"] for p in predictions])) if predictions else 0.0, 4)
        }

    def _generate_prometheus_metrics(self, perf_metrics: dict,
                                       model_metrics: dict,
                                       drift_score: float) -> str:
        """Generate Prometheus metrics in text format."""
        model_name = model_metrics.get("model_name", "unknown")
        auc = model_metrics.get("auc", 0)

        metrics = """# HELP veda_model_auc Model AUC-ROC score
# TYPE veda_model_auc gauge
veda_model_auc{{model="{model}"}} {auc}

# HELP veda_prediction_latency_ms Prediction latency in milliseconds
# TYPE veda_prediction_latency_ms summary
veda_prediction_latency_ms{{quantile="0.5",model="{model}"}} {p50}
veda_prediction_latency_ms{{quantile="0.95",model="{model}"}} {p95}
veda_prediction_latency_ms{{quantile="0.99",model="{model}"}} {p99}

# HELP veda_prediction_errors_total Total prediction errors
# TYPE veda_prediction_errors_total counter
veda_prediction_errors_total{{model="{model}"}} {errors}

# HELP veda_error_rate Prediction error rate
# TYPE veda_error_rate gauge
veda_error_rate{{model="{model}"}} {error_rate}

# HELP veda_data_drift_score Data drift score
# TYPE veda_data_drift_score gauge
veda_data_drift_score{{model="{model}"}} {drift}

# HELP veda_positive_prediction_rate Rate of positive predictions
# TYPE veda_positive_prediction_rate gauge
veda_positive_prediction_rate{{model="{model}"}} {pos_rate}
""".format(
            model=model_name,
            auc=auc,
            p50=perf_metrics.get("latency_p50_ms", 0),
            p95=perf_metrics.get("latency_p95_ms", 0),
            p99=perf_metrics.get("latency_p99_ms", 0),
            errors=perf_metrics.get("errors", 0),
            error_rate=perf_metrics.get("error_rate", 0),
            drift=drift_score,
            pos_rate=perf_metrics.get("positive_rate", 0)
        )
        return metrics

    def _check_alerts(self, perf_metrics: dict, model_metrics: dict,
                       drift_score: float) -> list:
        """Check for alert conditions."""
        alerts = []
        auc = model_metrics.get("auc", 1.0)

        if perf_metrics.get("latency_p99_ms", 0) > 100:
            alerts.append({
                "severity": "WARNING",
                "message": "P99 latency " + str(perf_metrics["latency_p99_ms"]) + "ms exceeds 100ms threshold"
            })

        if perf_metrics.get("error_rate", 0) > 0.01:
            alerts.append({
                "severity": "CRITICAL",
                "message": "Error rate " + str(perf_metrics["error_rate"]) + " exceeds 1% threshold"
            })

        if drift_score > 0.2:
            alerts.append({
                "severity": "WARNING",
                "message": "Data drift score " + str(drift_score) + " exceeds 0.2 threshold"
            })

        if auc < 0.7:
            alerts.append({
                "severity": "CRITICAL",
                "message": "Model AUC " + str(auc) + " below 0.7 threshold — consider retraining"
            })

        return alerts

    def run(self, state: dict) -> dict:
        """
        MLOps Monitoring:
        1. Load model and data
        2. Simulate production requests
        3. Track latency and errors
        4. Generate Prometheus metrics
        5. Check alert conditions
        """

        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        model_info = state.get("model_info", {})
        drift_score = state.get("drift_report", {}).get("drift_score", 0)

        self.log("Loading model and features...")
        model = self._load_model(state)
        df = self._load_features(state)

        if model is None or df is None:
            self.log("Model or features not found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        X = df.drop(columns=[target_col]).values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        # Simulate 200 prediction requests
        self.log("Simulating 200 production prediction requests...")
        perf_metrics = self._simulate_prediction_requests(model, X, n_requests=200)

        self.log("Mean latency   : " + str(perf_metrics["latency_mean_ms"]) + "ms")
        self.log("P95 latency    : " + str(perf_metrics["latency_p95_ms"]) + "ms")
        self.log("P99 latency    : " + str(perf_metrics["latency_p99_ms"]) + "ms")
        self.log("Error rate     : " + str(perf_metrics["error_rate"]))
        self.log("Positive rate  : " + str(perf_metrics["positive_rate"]))

        # Model metrics for monitoring
        model_metrics = {
            "model_name": model_info.get("model_name", "unknown"),
            "auc": model_info.get("test_metrics", {}).get("auc_roc", 0),
            "f1": model_info.get("test_metrics", {}).get("f1_score", 0)
        }

        # Generate Prometheus metrics
        self.log("Generating Prometheus metrics...")
        prometheus_metrics = self._generate_prometheus_metrics(
            perf_metrics, model_metrics, drift_score
        )

        # Check alerts
        alerts = self._check_alerts(perf_metrics, model_metrics, drift_score)
        if alerts:
            for alert in alerts:
                self.log(alert["severity"] + ": " + alert["message"], level="WARN")
        else:
            self.log("No alerts triggered — all metrics within thresholds")

        # Save results
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        prometheus_path = "outputs/" + run_id + "_metrics.prom"
        with open(prometheus_path, "w") as f:
            f.write(prometheus_metrics)

        monitor_results = {
            "performance_metrics": perf_metrics,
            "model_metrics": model_metrics,
            "drift_score": drift_score,
            "alerts": alerts,
            "prometheus_path": prometheus_path
        }

        results_path = "outputs/" + run_id + "_monitor_results.json"
        with open(results_path, "w") as f:
            json.dump(monitor_results, f, indent=2)

        state["monitor_results"] = monitor_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] MLOpsMonitorAgent: " +
            "p99=" + str(perf_metrics["latency_p99_ms"]) + "ms " +
            "errors=" + str(perf_metrics["errors"]) + " " +
            "alerts=" + str(len(alerts))
        )

        self.log("=" * 50)
        self.log("MLOPS MONITORING COMPLETE")
        self.log("Requests    : " + str(perf_metrics["total_requests"]))
        self.log("P99 latency : " + str(perf_metrics["latency_p99_ms"]) + "ms")
        self.log("Error rate  : " + str(perf_metrics["error_rate"]))
        self.log("Alerts      : " + str(len(alerts)))
        self.log("Prom metrics: " + prometheus_path)
        self.log("=" * 50)

        return state