"""
VEDA — Autonomous Data Science System
agents/streaming/stream_anomaly.py — Stream Anomaly Agent

Real-time anomaly detection on streams:
- Statistical control charts
- EWMA anomaly detection
- Isolation Forest online
- Alert generation
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from veda.core.base_agent import BaseAgent


class StreamAnomalyAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="StreamAnomalyAgent",
            domain="streaming",
            version="1.0.0"
        )

    def _load_stream_data(self, state):
        stream_data = state.get("stream_data")
        if stream_data is not None:
            return stream_data if isinstance(stream_data, pd.DataFrame) \
                   else pd.DataFrame(stream_data)
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_stream_data.parquet")]
        if files:
            return pd.read_parquet(os.path.join(d, sorted(files)[-1]))
        return None

    def _ewma_anomaly(self, series: np.ndarray,
                       alpha: float = 0.1,
                       threshold: float = 3.0) -> dict:
        """EWMA control chart anomaly detection."""
        ewma = np.zeros(len(series))
        ewma[0] = series[0]

        for i in range(1, len(series)):
            ewma[i] = alpha * series[i] + (1 - alpha) * ewma[i-1]

        residuals = series - ewma
        std = np.std(residuals)
        anomaly_mask = np.abs(residuals) > threshold * std
        anomaly_indices = list(np.where(anomaly_mask)[0])

        return {
            "method": "EWMA",
            "alpha": alpha,
            "threshold": threshold,
            "anomaly_count": int(anomaly_mask.sum()),
            "anomaly_rate": round(float(anomaly_mask.mean() * 100), 2),
            "anomaly_indices": anomaly_indices[:10],
            "anomaly_values": [round(float(series[i]), 4) for i in anomaly_indices[:10]]
        }

    def _isolation_forest_stream(self, df: pd.DataFrame,
                                  contamination: float = 0.05) -> dict:
        """Isolation Forest on stream data."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        X = df[numeric_cols].fillna(0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=50
        )
        predictions = iso.fit_predict(X_scaled)
        anomaly_mask = predictions == -1
        anomaly_scores = iso.score_samples(X_scaled)

        return {
            "method": "IsolationForest",
            "contamination": contamination,
            "anomaly_count": int(anomaly_mask.sum()),
            "anomaly_rate": round(float(anomaly_mask.mean() * 100), 2),
            "avg_anomaly_score": round(float(anomaly_scores[anomaly_mask].mean()), 4) if anomaly_mask.any() else 0,
            "anomaly_indices": list(np.where(anomaly_mask)[0])[:10]
        }

    def _statistical_process_control(self, series: np.ndarray) -> dict:
        """Statistical Process Control (SPC) with control limits."""
        mean = np.mean(series)
        std = np.std(series)
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        violations = np.where((series > ucl) | (series < lcl))[0]

        return {
            "method": "SPC",
            "mean": round(float(mean), 4),
            "std": round(float(std), 4),
            "ucl": round(float(ucl), 4),
            "lcl": round(float(lcl), 4),
            "violations": int(len(violations)),
            "violation_rate": round(float(len(violations) / len(series) * 100), 2),
            "violation_indices": list(violations)[:10]
        }

    def _generate_alerts(self, ewma: dict, iso: dict, spc: dict) -> list:
        """Generate alerts based on anomaly detection."""
        alerts = []

        if ewma["anomaly_rate"] > 5:
            alerts.append({
                "severity": "WARNING",
                "source": "EWMA",
                "message": "EWMA anomaly rate " + str(ewma["anomaly_rate"]) + "% exceeds 5% threshold"
            })

        if iso["anomaly_rate"] > 8:
            alerts.append({
                "severity": "CRITICAL",
                "source": "IsolationForest",
                "message": "Isolation Forest detected " + str(iso["anomaly_count"]) + " anomalies"
            })

        if spc["violation_rate"] > 3:
            alerts.append({
                "severity": "WARNING",
                "source": "SPC",
                "message": "SPC violations: " + str(spc["violations"]) + " points outside control limits"
            })

        return alerts

    def run(self, state: dict) -> dict:
        self.log("Loading stream data for anomaly detection...")
        df = self._load_stream_data(state)

        if df is None:
            self.log("No stream data found", level="WARN")
            return state

        self.log("Stream data: " + str(df.shape))

        primary_col = "value" if "value" in df.columns else \
                     df.select_dtypes(include=[np.number]).columns[0]
        series = df[primary_col].fillna(df[primary_col].mean()).values

        self.log("Running EWMA anomaly detection...")
        ewma = self._ewma_anomaly(series, alpha=0.1, threshold=3.0)
        self.log("EWMA anomalies: " + str(ewma["anomaly_count"]) +
                " (" + str(ewma["anomaly_rate"]) + "%)")

        self.log("Running Isolation Forest...")
        iso = self._isolation_forest_stream(df)
        self.log("ISO anomalies: " + str(iso["anomaly_count"]) +
                " (" + str(iso["anomaly_rate"]) + "%)")

        self.log("Running Statistical Process Control...")
        spc = self._statistical_process_control(series)
        self.log("SPC violations: " + str(spc["violations"]))

        self.log("Generating alerts...")
        alerts = self._generate_alerts(ewma, iso, spc)
        for alert in alerts:
            self.log(alert["severity"] + ": " + alert["message"], level="WARN")

        stream_anomaly_results = {
            "primary_metric": primary_col,
            "ewma": ewma,
            "isolation_forest": iso,
            "spc": spc,
            "alerts": alerts,
            "total_anomalies": max(ewma["anomaly_count"], iso["anomaly_count"])
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_stream_anomaly.json"
        with open(path, "w") as f:
            json.dump(stream_anomaly_results, f, indent=2)

        state["stream_anomaly"] = stream_anomaly_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] StreamAnomalyAgent: " +
            "EWMA=" + str(ewma["anomaly_count"]) +
            " ISO=" + str(iso["anomaly_count"]) +
            " alerts=" + str(len(alerts))
        )

        self.log("=" * 50)
        self.log("STREAM ANOMALY DETECTION COMPLETE")
        self.log("EWMA anomalies : " + str(ewma["anomaly_count"]))
        self.log("ISO anomalies  : " + str(iso["anomaly_count"]))
        self.log("SPC violations : " + str(spc["violations"]))
        self.log("Alerts         : " + str(len(alerts)))
        self.log("=" * 50)

        return state