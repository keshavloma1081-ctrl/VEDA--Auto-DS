"""
VEDA — Autonomous Data Science System
agents/aiops/auto_healing.py — Auto Healing Agent

Automatically suggests and applies fixes:
- Memory optimization
- Connection pool tuning
- Cache clearing
- Service restart recommendations
- Auto-scaling triggers
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class AutoHealingAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="AutoHealingAgent",
            domain="aiops",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

        self.healing_playbooks = {
            "OutOfMemory": {
                "action": "MEMORY_OPTIMIZATION",
                "steps": [
                    "Clear model cache",
                    "Reduce batch size from 512 to 256",
                    "Trigger garbage collection",
                    "Restart service if memory > 95%"
                ],
                "auto_apply": True,
                "risk": "LOW"
            },
            "ConnectionError": {
                "action": "CONNECTION_RESET",
                "steps": [
                    "Reset connection pool",
                    "Increase connection timeout to 60s",
                    "Enable connection retry with exponential backoff",
                    "Check database health"
                ],
                "auto_apply": True,
                "risk": "LOW"
            },
            "Timeout": {
                "action": "TIMEOUT_MITIGATION",
                "steps": [
                    "Increase request timeout threshold",
                    "Enable circuit breaker",
                    "Route traffic to backup instance",
                    "Scale up service replicas"
                ],
                "auto_apply": False,
                "risk": "MEDIUM"
            },
            "DiskFull": {
                "action": "DISK_CLEANUP",
                "steps": [
                    "Archive old log files",
                    "Remove temporary files",
                    "Clean ML model artifacts older than 30 days",
                    "Alert storage team"
                ],
                "auto_apply": True,
                "risk": "LOW"
            },
            "ModelError": {
                "action": "MODEL_ROLLBACK",
                "steps": [
                    "Load previous stable model version",
                    "Disable new model endpoint",
                    "Alert ML team",
                    "Trigger model revalidation"
                ],
                "auto_apply": False,
                "risk": "HIGH"
            }
        }

    def _select_healing_actions(self, rca_report: dict,
                                 anomaly_results: dict) -> list:
        """Select appropriate healing actions based on RCA."""
        selected_actions = []
        rca = rca_report.get("root_cause_analysis", {})
        error_patterns = rca_report.get("error_patterns", [])
        alerts = anomaly_results.get("alerts", [])

        for error_type in error_patterns:
            if error_type in self.healing_playbooks:
                playbook = self.healing_playbooks[error_type]
                selected_actions.append({
                    "trigger": error_type,
                    "action": playbook["action"],
                    "steps": playbook["steps"],
                    "auto_apply": playbook["auto_apply"],
                    "risk": playbook["risk"],
                    "status": "PENDING"
                })

        if not selected_actions and alerts:
            selected_actions.append({
                "trigger": "GENERAL_DEGRADATION",
                "action": "SYSTEM_HEALTH_CHECK",
                "steps": [
                    "Run full system health check",
                    "Review all service logs",
                    "Check resource utilization",
                    "Notify on-call engineer"
                ],
                "auto_apply": False,
                "risk": "LOW",
                "status": "PENDING"
            })

        return selected_actions

    def _simulate_healing(self, actions: list) -> list:
        """Simulate applying healing actions."""
        results = []
        for action in actions:
            if action["auto_apply"] and action["risk"] == "LOW":
                status = "APPLIED"
                message = "Auto-applied successfully"
            elif action["risk"] == "HIGH":
                status = "REQUIRES_APPROVAL"
                message = "High risk — requires human approval before applying"
            else:
                status = "RECOMMENDED"
                message = "Manual application recommended"

            results.append({
                **action,
                "status": status,
                "message": message,
                "applied_at": datetime.now().isoformat() if status == "APPLIED" else None
            })

        return results

    def _generate_healing_summary(self, results: list, rca: dict) -> str:
        """Generate healing summary with Groq."""
        applied = [r for r in results if r["status"] == "APPLIED"]
        pending = [r for r in results if r["status"] != "APPLIED"]

        prompt = """You are an SRE. Summarize these auto-healing actions.

Root cause: """ + str(rca.get("root_cause_analysis", {}).get("primary_root_cause", "Unknown")) + """

Actions applied: """ + str(len(applied)) + """
Actions pending: """ + str(len(pending)) + """

Applied actions: """ + json.dumps([r["action"] for r in applied]) + """
Pending actions: """ + json.dumps([r["action"] for r in pending]) + """

Write 2-3 sentences summarizing what was fixed and what still needs attention."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except:
            return str(len(applied)) + " healing actions applied automatically. " + str(len(pending)) + " actions require manual review."

    def run(self, state: dict) -> dict:
        """
        Auto Healing:
        1. Load RCA and anomaly data
        2. Select healing actions
        3. Simulate applying actions
        4. Generate healing report
        """

        rca_report = state.get("rca_report", {})
        anomaly_results = state.get("anomaly_results", {})

        if not rca_report and not anomaly_results:
            self.log("No RCA or anomaly data found", level="WARN")
            return state

        self.log("Selecting healing actions from playbooks...")
        actions = self._select_healing_actions(rca_report, anomaly_results)
        self.log("Selected " + str(len(actions)) + " healing actions")

        self.log("Simulating healing actions...")
        results = self._simulate_healing(actions)

        applied = [r for r in results if r["status"] == "APPLIED"]
        pending = [r for r in results if r["status"] != "APPLIED"]

        self.log("Applied: " + str(len(applied)))
        self.log("Pending: " + str(len(pending)))

        for r in results:
            self.log(r["action"] + " -> " + r["status"] + ": " + r["message"])

        self.log("Generating healing summary...")
        summary = self._generate_healing_summary(results, rca_report)

        healing_report = {
            "total_actions": len(results),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "actions": results,
            "summary": summary
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_healing_report.json"
        with open(path, "w") as f:
            json.dump(healing_report, f, indent=2)

        state["healing_report"] = healing_report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] AutoHealingAgent: " +
            str(len(applied)) + " applied, " +
            str(len(pending)) + " pending"
        )

        self.log("=" * 50)
        self.log("AUTO HEALING COMPLETE")
        self.log("Applied : " + str(len(applied)))
        self.log("Pending : " + str(len(pending)))
        self.log("Summary : " + summary[:150])
        self.log("=" * 50)

        return state