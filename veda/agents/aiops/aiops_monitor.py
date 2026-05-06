"""
VEDA — Autonomous Data Science System
agents/aiops/aiops_monitor.py — AIOps Monitor Agent

Continuous system monitoring dashboard:
- Real-time metric summary
- SLA compliance check
- Incident timeline
- System health score
- Monitoring report
"""

import os
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class AIOpsMonitorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="AIOpsMonitorAgent",
            domain="aiops",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

        self.sla_thresholds = {
            "availability": 99.9,
            "latency_p99_ms": 200,
            "error_rate_pct": 1.0,
            "cpu_pct": 80,
            "memory_pct": 85
        }

    def _compute_sla_compliance(self, anomaly_results: dict) -> dict:
        """Check SLA compliance against thresholds."""
        compliance = {}
        anomaly_details = anomaly_results.get("anomaly_details", {})

        for metric, threshold_key in [
            ("cpu_pct", "cpu_pct"),
            ("memory_pct", "memory_pct"),
            ("latency_ms", "latency_p99_ms"),
            ("error_rate", "error_rate_pct")
        ]:
            if metric in anomaly_details:
                details = anomaly_details[metric]
                max_val = details.get("max", 0)
                threshold = self.sla_thresholds.get(threshold_key, 100)
                breached = max_val > threshold
                compliance[metric] = {
                    "threshold": threshold,
                    "max_observed": round(float(max_val), 2),
                    "mean_observed": round(float(details.get("mean", 0)), 2),
                    "breached": bool(breached),
                    "anomaly_count": details.get("anomaly_count", 0)
                }

        breached_count = sum(1 for v in compliance.values() if v["breached"])
        total = len(compliance)
        sla_score = round((total - breached_count) / total * 100, 2) if total > 0 else 100

        return {
            "metrics": compliance,
            "sla_score": sla_score,
            "breached_count": breached_count,
            "status": "COMPLIANT" if breached_count == 0 else "BREACHED"
        }

    def _compute_health_score(self, log_analysis: dict,
                               anomaly_results: dict,
                               healing_report: dict) -> dict:
        """Compute overall system health score."""
        scores = {}

        log_health = log_analysis.get("error_rate", {}).get("health_score", 100)
        scores["log_health"] = round(float(log_health), 2)

        total_anomalies = anomaly_results.get("total_anomalies", 0)
        anomaly_health = max(0, 100 - total_anomalies * 2)
        scores["anomaly_health"] = round(float(anomaly_health), 2)

        critical_alerts = anomaly_results.get("critical_alerts", 0)
        alert_health = max(0, 100 - critical_alerts * 20)
        scores["alert_health"] = round(float(alert_health), 2)

        applied = healing_report.get("applied_count", 0)
        pending = healing_report.get("pending_count", 0)
        healing_health = 100 if applied + pending == 0 else round(applied / max(applied + pending, 1) * 100, 2)
        scores["healing_health"] = healing_health

        overall = round(np.mean(list(scores.values())), 2)

        return {
            "component_scores": scores,
            "overall_score": overall,
            "grade": "A" if overall >= 90 else "B" if overall >= 75 else "C" if overall >= 60 else "D"
        }

    def _generate_monitoring_report(self, health: dict, sla: dict,
                                     anomalies: dict, healing: dict) -> str:
        """Generate monitoring report with Groq."""
        prompt = """You are an SRE generating a system monitoring report.

Health Score: """ + str(health["overall_score"]) + "/100 (Grade: " + health["grade"] + """)"
SLA Status: """ + sla["status"] + " (score=" + str(sla["sla_score"]) + """%)
Total Anomalies: """ + str(anomalies.get("total_anomalies", 0)) + """
Healing Actions Applied: """ + str(healing.get("applied_count", 0)) + """
Healing Actions Pending: """ + str(healing.get("pending_count", 0)) + """

Write a 3-4 sentence executive monitoring summary covering:
1. Current system health status
2. Key issues identified
3. Actions taken and outstanding items
4. Overall risk assessment"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except:
            return "System health score: " + str(health["overall_score"]) + "/100. " + str(anomalies.get("total_anomalies", 0)) + " anomalies detected."

    def _generate_html_dashboard(self, health: dict, sla: dict,
                                  anomalies: dict, healing: dict,
                                  report: str, run_id: str) -> str:
        """Generate HTML monitoring dashboard."""
        grade = health["grade"]
        score = health["overall_score"]
        score_color = "#2e7d32" if score >= 75 else "#e65100" if score >= 50 else "#c62828"

        html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>VEDA AIOps Dashboard</title>
<style>
body{font-family:Arial,sans-serif;margin:30px;color:#1a1a2e;background:#f5f5f5;}
.header{background:#1F4E79;color:white;padding:20px;border-radius:10px;margin-bottom:20px;}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
.card{background:white;padding:15px;border-radius:8px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
.card .val{font-size:28px;font-weight:bold;}
.card .lbl{font-size:11px;color:#666;margin-top:4px;}
.section{background:white;padding:18px;border-radius:8px;margin-bottom:15px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
.section h3{color:#1F4E79;margin:0 0 12px;}
table{width:100%;border-collapse:collapse;}
th{background:#1F4E79;color:white;padding:8px;text-align:left;font-size:12px;}
td{padding:7px 8px;border-bottom:1px solid #eee;font-size:12px;}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:bold;}
.summary{background:#e3f2fd;padding:15px;border-radius:8px;border-left:4px solid #1F4E79;}
</style>
</head>
<body>
<div class="header">
<h2 style="margin:0">VEDA AIOps Monitoring Dashboard</h2>
<p style="margin:5px 0 0;opacity:.85">Run ID: """ + run_id + """ | Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
</div>

<div class="grid">
<div class="card">
<div class="val" style="color:""" + score_color + """">""" + str(score) + """</div>
<div class="lbl">Health Score / 100</div>
</div>
<div class="card">
<div class="val" style="color:""" + score_color + """">""" + grade + """</div>
<div class="lbl">Health Grade</div>
</div>
<div class="card">
<div class="val">""" + str(anomalies.get("total_anomalies", 0)) + """</div>
<div class="lbl">Total Anomalies</div>
</div>
<div class="card">
<div class="val">""" + str(healing.get("applied_count", 0)) + """</div>
<div class="lbl">Auto-Healed</div>
</div>
</div>

<div class="section">
<h3>Executive Summary</h3>
<div class="summary">""" + report.replace("\n", "<br>") + """</div>
</div>

<div class="section">
<h3>SLA Compliance — """ + sla["status"] + """ (""" + str(sla["sla_score"]) + """%)</h3>
<table>
<tr><th>Metric</th><th>Threshold</th><th>Max Observed</th><th>Mean</th><th>Status</th></tr>
""" + "".join([
    "<tr><td>" + metric + "</td><td>" + str(v["threshold"]) + "</td><td>" +
    str(v["max_observed"]) + "</td><td>" + str(v["mean_observed"]) + "</td><td>" +
    "<span class='badge' style='background:" + ("#ffebee" if v["breached"] else "#e8f5e9") +
    ";color:" + ("#c62828" if v["breached"] else "#2e7d32") + "'>" +
    ("BREACHED" if v["breached"] else "OK") + "</span></td></tr>"
    for metric, v in sla.get("metrics", {}).items()
]) + """
</table>
</div>

<div class="section">
<h3>Component Health Scores</h3>
<table>
<tr><th>Component</th><th>Score</th></tr>
""" + "".join([
    "<tr><td>" + k.replace("_", " ").title() + "</td><td>" + str(v) + "/100</td></tr>"
    for k, v in health.get("component_scores", {}).items()
]) + """
</table>
</div>

</body>
</html>"""
        return html

    def run(self, state: dict) -> dict:
        """
        AIOps Monitor:
        1. Compute SLA compliance
        2. Calculate health score
        3. Generate monitoring report
        4. Create HTML dashboard
        """

        log_analysis = state.get("log_analysis", {})
        anomaly_results = state.get("anomaly_results", {})
        healing_report = state.get("healing_report", {})

        self.log("Computing SLA compliance...")
        sla = self._compute_sla_compliance(anomaly_results)
        self.log("SLA status: " + sla["status"] + " (score=" + str(sla["sla_score"]) + "%)")

        self.log("Computing system health score...")
        health = self._compute_health_score(log_analysis, anomaly_results, healing_report)
        self.log("Health score: " + str(health["overall_score"]) + "/100 (Grade: " + health["grade"] + ")")

        self.log("Generating monitoring report...")
        report = self._generate_monitoring_report(health, sla, anomaly_results, healing_report)

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.log("Building HTML dashboard...")
        html = self._generate_html_dashboard(health, sla, anomaly_results, healing_report, report, run_id)
        dashboard_path = "outputs/" + run_id + "_aiops_dashboard.html"
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html)

        monitor_results = {
            "health": health,
            "sla": sla,
            "report": report,
            "dashboard_path": dashboard_path
        }

        results_path = "outputs/" + run_id + "_aiops_monitor.json"
        with open(results_path, "w") as f:
            json.dump(monitor_results, f, indent=2)

        state["aiops_monitor"] = monitor_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] AIOpsMonitorAgent: " +
            "health=" + str(health["overall_score"]) +
            " grade=" + health["grade"] +
            " sla=" + sla["status"]
        )

        self.log("=" * 50)
        self.log("AIOPS MONITORING COMPLETE")
        self.log("Health score : " + str(health["overall_score"]) + "/100")
        self.log("Grade        : " + health["grade"])
        self.log("SLA status   : " + sla["status"])
        self.log("Dashboard    : " + dashboard_path)
        self.log("=" * 50)

        return state