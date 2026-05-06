"""
VEDA — Autonomous Data Science System
agents/aiops/root_cause.py — Root Cause Analysis Agent

Identifies root causes of system issues:
- Correlates anomalies with log errors
- Builds causal chain
- LLM-powered root cause analysis
- Generates incident report
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class RootCauseAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="RootCauseAgent",
            domain="aiops",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _correlate_anomalies_with_logs(self, anomaly_results: dict,
                                        log_analysis: dict) -> list:
        """Find correlations between metric anomalies and log errors."""
        correlations = []
        alerts = anomaly_results.get("alerts", [])
        error_patterns = log_analysis.get("error_patterns", {})

        correlation_map = {
            "cpu_pct": ["OutOfMemory", "Timeout"],
            "memory_pct": ["OutOfMemory", "NullPointer"],
            "latency_ms": ["Timeout", "ConnectionError", "DatabaseError"],
            "error_rate": ["ModelError", "ConnectionError", "PermissionDenied"]
        }

        for alert in alerts:
            metric = alert["metric"]
            related_errors = correlation_map.get(metric, [])
            for error_type in related_errors:
                if error_type in error_patterns:
                    correlations.append({
                        "metric": metric,
                        "metric_value": alert["value"],
                        "related_error": error_type,
                        "error_count": error_patterns[error_type]["count"],
                        "confidence": "HIGH" if alert["severity"] == "CRITICAL" else "MEDIUM"
                    })

        return correlations

    def _build_causal_chain(self, correlations: list,
                             error_patterns: dict) -> list:
        """Build causal chain from correlations."""
        if not correlations:
            return ["No clear causal chain identified — system appears stable"]

        chain = []
        seen = set()

        for corr in correlations[:5]:
            cause = corr["related_error"]
            effect = corr["metric"]

            if cause not in seen:
                seen.add(cause)
                chain.append(
                    cause + " (" + str(corr["error_count"]) + " occurrences)" +
                    " -> " + effect + " degradation" +
                    " [" + corr["confidence"] + " confidence]"
                )

        return chain

    def _llm_root_cause_analysis(self, anomalies: dict,
                                  logs: dict, correlations: list,
                                  causal_chain: list) -> dict:
        """Use Groq LLM for root cause analysis."""
        prompt = """You are an expert Site Reliability Engineer performing root cause analysis.

System Alerts:
""" + json.dumps(anomalies.get("alerts", [])[:5], indent=2) + """

Log Error Patterns:
""" + json.dumps(logs.get("error_patterns", {}), indent=2) + """

Detected Correlations:
""" + json.dumps(correlations[:5], indent=2) + """

Perform root cause analysis and provide:
1. Primary root cause (most likely cause)
2. Contributing factors
3. Impact assessment
4. Immediate remediation steps

Return JSON only:
{
    "primary_root_cause": "...",
    "contributing_factors": ["...", "..."],
    "impact": "...",
    "remediation_steps": ["...", "..."],
    "confidence": "HIGH/MEDIUM/LOW"
}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an SRE expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            return {
                "primary_root_cause": "Unable to determine — insufficient data",
                "contributing_factors": ["Check system logs for more details"],
                "impact": "Unknown",
                "remediation_steps": ["Review error logs", "Check system resources"],
                "confidence": "LOW"
            }

    def run(self, state: dict) -> dict:
        """
        Root Cause Analysis:
        1. Load anomalies and logs
        2. Correlate anomalies with logs
        3. Build causal chain
        4. LLM root cause analysis
        5. Generate incident report
        """

        anomaly_results = state.get("anomaly_results", {})
        log_analysis = state.get("log_analysis", {})

        if not anomaly_results and not log_analysis:
            self.log("No anomaly or log data found", level="WARN")
            return state

        self.log("Correlating anomalies with log errors...")
        correlations = self._correlate_anomalies_with_logs(
            anomaly_results, log_analysis
        )
        self.log("Found " + str(len(correlations)) + " correlations")

        self.log("Building causal chain...")
        causal_chain = self._build_causal_chain(
            correlations,
            log_analysis.get("error_patterns", {})
        )
        for step in causal_chain:
            self.log("  -> " + step)

        self.log("Running LLM root cause analysis...")
        rca = self._llm_root_cause_analysis(
            anomaly_results, log_analysis, correlations, causal_chain
        )

        self.log("Primary root cause: " + rca.get("primary_root_cause", "Unknown"))
        self.log("Confidence: " + rca.get("confidence", "Unknown"))

        incident_report = {
            "timestamp": datetime.now().isoformat(),
            "correlations": correlations,
            "causal_chain": causal_chain,
            "root_cause_analysis": rca,
            "alert_count": anomaly_results.get("alert_count", 0),
            "error_patterns": list(log_analysis.get("error_patterns", {}).keys()),
            "incident_severity": "CRITICAL" if anomaly_results.get("critical_alerts", 0) > 0 else "WARNING"
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_rca_report.json"
        with open(path, "w") as f:
            json.dump(incident_report, f, indent=2)

        state["rca_report"] = incident_report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] RootCauseAgent: " +
            rca.get("primary_root_cause", "unknown")[:50] +
            " confidence=" + rca.get("confidence", "LOW")
        )

        self.log("=" * 50)
        self.log("ROOT CAUSE ANALYSIS COMPLETE")
        self.log("Root cause  : " + rca.get("primary_root_cause", "Unknown")[:80])
        self.log("Confidence  : " + rca.get("confidence", "LOW"))
        self.log("Remediation : " + str(rca.get("remediation_steps", [])[:2]))
        self.log("=" * 50)

        return state