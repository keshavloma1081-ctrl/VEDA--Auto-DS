"""
VEDA — Autonomous Data Science System
agents/aiops/anomaly_detection.py — Anomaly Detection Agent
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

from veda.core.base_agent import BaseAgent


def np_convert(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    raise TypeError


class AnomalyDetectionAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="AnomalyDetectionAgent",
            domain="aiops",
            version="1.0.0"
        )
        self.z_threshold = 3.0
        self.iqr_multiplier = 1.5

    def _generate_metrics(self, n_points=200):
        np.random.seed(42)
        timestamps = [
            datetime.now() - timedelta(minutes=i)
            for i in range(n_points, 0, -1)
        ]
        cpu = np.random.normal(45, 10, n_points)
        memory = np.random.normal(60, 8, n_points)
        latency = np.random.normal(50, 15, n_points)
        error_rate = np.random.normal(2, 1, n_points)
        throughput = np.random.normal(1000, 100, n_points)

        for idx in [50, 100, 150, 175]:
            if idx < n_points:
                cpu[idx] = 95 + np.random.uniform(0, 5)
                memory[idx] = 92 + np.random.uniform(0, 5)
                latency[idx] = 500 + np.random.uniform(0, 100)
                error_rate[idx] = 25 + np.random.uniform(0, 10)

        return pd.DataFrame({
            "timestamp": timestamps,
            "cpu_pct": np.clip(cpu, 0, 100),
            "memory_pct": np.clip(memory, 0, 100),
            "latency_ms": np.clip(latency, 0, None),
            "error_rate": np.clip(error_rate, 0, 100),
            "throughput": np.clip(throughput, 0, None)
        })

    def _zscore_anomalies(self, series):
        z_scores = np.abs(stats.zscore(series))
        return [int(i) for i in np.where(z_scores > self.z_threshold)[0]]

    def _iqr_anomalies(self, series):
        Q1 = np.percentile(series, 25)
        Q3 = np.percentile(series, 75)
        IQR = Q3 - Q1
        lower = Q1 - self.iqr_multiplier * IQR
        upper = Q3 + self.iqr_multiplier * IQR
        return [int(i) for i in np.where((series < lower) | (series > upper))[0]]

    def _detect_metric_anomalies(self, df):
        metric_cols = [c for c in df.columns if c != "timestamp"]
        results = {}
        for col in metric_cols:
            series = df[col].values
            z_anomalies = self._zscore_anomalies(series)
            iqr_anomalies = self._iqr_anomalies(series)
            combined = list(set(z_anomalies + iqr_anomalies))
            results[col] = {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "anomaly_count": int(len(combined)),
                "anomaly_indices": combined[:10],
                "anomaly_values": [round(float(series[i]), 2) for i in combined[:5]],
                "anomaly_pct": round(float(len(combined) / len(series) * 100), 2)
            }
        return results

    def _generate_alerts(self, anomalies, df):
        alerts = []
        latest = df.iloc[-1]

        if float(latest["cpu_pct"]) > 90:
            alerts.append({
                "severity": "CRITICAL",
                "metric": "cpu_pct",
                "value": round(float(latest["cpu_pct"]), 2),
                "message": "CPU usage " + str(round(float(latest["cpu_pct"]), 1)) + "% exceeds 90% threshold"
            })
        if float(latest["memory_pct"]) > 85:
            alerts.append({
                "severity": "WARNING",
                "metric": "memory_pct",
                "value": round(float(latest["memory_pct"]), 2),
                "message": "Memory usage " + str(round(float(latest["memory_pct"]), 1)) + "% exceeds 85%"
            })
        if float(latest["latency_ms"]) > 200:
            alerts.append({
                "severity": "WARNING",
                "metric": "latency_ms",
                "value": round(float(latest["latency_ms"]), 2),
                "message": "Latency " + str(round(float(latest["latency_ms"]), 1)) + "ms exceeds 200ms"
            })
        if float(latest["error_rate"]) > 5:
            alerts.append({
                "severity": "CRITICAL",
                "metric": "error_rate",
                "value": round(float(latest["error_rate"]), 2),
                "message": "Error rate " + str(round(float(latest["error_rate"]), 1)) + "% exceeds 5%"
            })
        for col, result in anomalies.items():
            if result["anomaly_count"] > 5:
                alerts.append({
                    "severity": "WARNING",
                    "metric": col,
                    "value": result["anomaly_count"],
                    "message": col + " has " + str(result["anomaly_count"]) + " anomalies"
                })
        return alerts

    def run(self, state: dict) -> dict:
        self.log("Generating system metrics...")
        df = self._generate_metrics(n_points=200)
        self.log("Metrics shape: " + str(df.shape))

        self.log("Detecting anomalies...")
        anomalies = self._detect_metric_anomalies(df)

        total_anomalies = int(sum(r["anomaly_count"] for r in anomalies.values()))
        self.log("Total anomalies: " + str(total_anomalies))

        for metric, result in anomalies.items():
            if result["anomaly_count"] > 0:
                self.log(metric + ": " + str(result["anomaly_count"]) + " anomalies")

        self.log("Generating alerts...")
        alerts = self._generate_alerts(anomalies, df)
        for alert in alerts:
            self.log(alert["severity"] + ": " + alert["message"], level="WARN")

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        anomaly_results = {
            "metrics_analyzed": list(anomalies.keys()),
            "total_anomalies": total_anomalies,
            "anomaly_details": anomalies,
            "alerts": alerts,
            "alert_count": int(len(alerts)),
            "critical_alerts": int(sum(1 for a in alerts if a["severity"] == "CRITICAL"))
        }

        path = "outputs/" + run_id + "_anomaly_results.json"
        with open(path, "w") as f:
            json.dump(anomaly_results, f, indent=2, default=np_convert)

        state["anomaly_results"] = anomaly_results
        state["system_metrics"] = df.head(10).to_dict(orient="records")
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] AnomalyDetectionAgent: " +
            str(total_anomalies) + " anomalies, " + str(len(alerts)) + " alerts"
        )

        self.log("=" * 50)
        self.log("ANOMALY DETECTION COMPLETE")
        self.log("Total anomalies : " + str(total_anomalies))
        self.log("Alerts          : " + str(len(alerts)))
        self.log("Critical alerts : " + str(anomaly_results["critical_alerts"]))
        self.log("=" * 50)

        return state