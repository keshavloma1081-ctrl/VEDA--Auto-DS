"""
VEDA — Autonomous Data Science System
agents/streaming/stream_monitor.py — Stream Monitor Agent

Monitors streaming pipeline health:
- Throughput metrics
- Latency tracking
- Consumer lag
- SLA compliance
- Dashboard generation
"""

import os
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class StreamMonitorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="StreamMonitorAgent",
            domain="streaming",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

        self.sla = {
            "max_latency_sec": 5.0,
            "min_throughput_eps": 50,
            "max_failure_rate_pct": 5.0,
            "max_anomaly_rate_pct": 10.0
        }

    def _compute_pipeline_health(self, state: dict) -> dict:
        """Compute overall streaming pipeline health."""
        ingest = state.get("stream_ingest", {})
        processor = state.get("stream_processor", {})
        anomaly = state.get("stream_anomaly", {})
        online = state.get("online_learning", {})

        scores = {}

        throughput = ingest.get("throughput_per_sec", 0)
        scores["throughput"] = round(min(throughput / self.sla["min_throughput_eps"] * 100, 100), 2)

        failure_rate = ingest.get("failure_rate", 100)
        scores["reliability"] = round(max(0, 100 - failure_rate * 10), 2)

        lag = ingest.get("lag_metrics", {}).get("avg_lag_sec", 100)
        scores["latency"] = round(max(0, 100 - (lag / self.sla["max_latency_sec"]) * 100), 2)

        anomaly_rate = anomaly.get("ewma", {}).get("anomaly_rate", 100)
        scores["data_quality"] = round(max(0, 100 - anomaly_rate * 5), 2)

        online_acc = online.get("hoeffding_tree", {}).get("final_accuracy", 0)
        scores["model_performance"] = round(float(online_acc) * 100, 2)

        overall = round(float(np.mean(list(scores.values()))), 2)

        return {
            "component_scores": scores,
            "overall_score": overall,
            "grade": "A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 55 else "D"
        }

    def _check_sla_compliance(self, state: dict) -> dict:
        """Check SLA compliance."""
        ingest = state.get("stream_ingest", {})
        anomaly = state.get("stream_anomaly", {})

        checks = {}

        throughput = ingest.get("throughput_per_sec", 0)
        checks["throughput"] = {
            "threshold": self.sla["min_throughput_eps"],
            "actual": round(float(throughput), 2),
            "passed": bool(throughput >= self.sla["min_throughput_eps"])
        }

        failure_rate = ingest.get("failure_rate", 100)
        checks["failure_rate"] = {
            "threshold": self.sla["max_failure_rate_pct"],
            "actual": round(float(failure_rate), 2),
            "passed": bool(failure_rate <= self.sla["max_failure_rate_pct"])
        }

        lag = ingest.get("lag_metrics", {}).get("avg_lag_sec", 100)
        checks["latency"] = {
            "threshold": self.sla["max_latency_sec"],
            "actual": round(float(lag), 2),
            "passed": bool(lag <= self.sla["max_latency_sec"])
        }

        anomaly_rate = anomaly.get("ewma", {}).get("anomaly_rate", 100)
        checks["anomaly_rate"] = {
            "threshold": self.sla["max_anomaly_rate_pct"],
            "actual": round(float(anomaly_rate), 2),
            "passed": bool(anomaly_rate <= self.sla["max_anomaly_rate_pct"])
        }

        passed = sum(1 for v in checks.values() if v["passed"])
        return {
            "checks": checks,
            "passed": passed,
            "total": len(checks),
            "sla_score": round(passed / len(checks) * 100, 1),
            "status": "COMPLIANT" if passed == len(checks) else "BREACHED"
        }

    def _generate_stream_summary(self, health: dict,
                                  sla: dict, state: dict) -> str:
        """Generate streaming summary."""
        ingest = state.get("stream_ingest", {})
        online = state.get("online_learning", {})

        prompt = """Summarize this streaming pipeline status in 3 sentences.

Health score: """ + str(health["overall_score"]) + "/100 (Grade: " + health["grade"] + """)"
SLA status: """ + sla["status"] + """
Throughput: """ + str(ingest.get("throughput_per_sec", 0)) + """ events/sec
Online model accuracy: """ + str(online.get("hoeffding_tree", {}).get("final_accuracy", 0)) + """
Concept drift: """ + str(online.get("concept_drift", {}).get("drift_detected", False)) + """

Cover: pipeline health, key issues, recommendations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except:
            return "Streaming pipeline health: " + str(health["overall_score"]) + "/100. SLA: " + sla["status"]

    def run(self, state: dict) -> dict:
        self.log("Computing pipeline health...")
        health = self._compute_pipeline_health(state)
        self.log("Health: " + str(health["overall_score"]) + "/100 Grade=" + health["grade"])

        self.log("Checking SLA compliance...")
        sla = self._check_sla_compliance(state)
        self.log("SLA: " + sla["status"] + " (" + str(sla["sla_score"]) + "%)")

        self.log("Generating summary...")
        summary = self._generate_stream_summary(health, sla, state)

        monitor_results = {
            "health": health,
            "sla": sla,
            "summary": summary,
            "pipeline_stats": {
                "events_ingested": state.get("stream_ingest", {}).get("successfully_ingested", 0),
                "throughput_eps": state.get("stream_ingest", {}).get("throughput_per_sec", 0),
                "tumbling_windows": state.get("stream_processor", {}).get("tumbling_windows", 0),
                "anomalies_detected": state.get("stream_anomaly", {}).get("total_anomalies", 0),
                "online_model": state.get("online_learning", {}).get("best_model", "N/A"),
                "online_accuracy": state.get("online_learning", {}).get(
                    "hoeffding_tree", {}
                ).get("final_accuracy", 0)
            }
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_stream_monitor.json"
        with open(path, "w") as f:
            json.dump(monitor_results, f, indent=2)

        state["stream_monitor"] = monitor_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] StreamMonitorAgent: " +
            "health=" + str(health["overall_score"]) +
            " grade=" + health["grade"] +
            " sla=" + sla["status"]
        )

        self.log("=" * 50)
        self.log("STREAM MONITOR COMPLETE")
        self.log("Health score : " + str(health["overall_score"]) + "/100")
        self.log("Grade        : " + health["grade"])
        self.log("SLA          : " + sla["status"])
        self.log("=" * 50)

        return state