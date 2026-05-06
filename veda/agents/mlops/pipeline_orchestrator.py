"""
VEDA — Autonomous Data Science System
agents/mlops/pipeline_orchestrator.py — Pipeline Orchestrator Agent

Manages VEDA pipeline scheduling:
- Schedule periodic runs
- Track run history
- Manage run configs
- Generate run reports
"""

import os
import json
from datetime import datetime, timedelta
from veda.core.base_agent import BaseAgent


class PipelineOrchestratorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="PipelineOrchestratorAgent",
            domain="mlops",
            version="1.0.0"
        )

    def _load_run_history(self) -> list:
        """Load previous run history."""
        history_path = "outputs/run_history.json"
        if os.path.exists(history_path):
            with open(history_path) as f:
                return json.load(f)
        return []

    def _save_run_history(self, history: list):
        """Save run history."""
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/run_history.json", "w") as f:
            json.dump(history, f, indent=2)

    def _generate_schedule_config(self, state: dict) -> dict:
        """Generate pipeline schedule configuration."""
        model_info = state.get("model_info", {})
        drift_report = state.get("drift_report", {})

        drift_score = drift_report.get("drift_score", 0)
        auc = model_info.get("test_metrics", {}).get("auc_roc", 0)

        if drift_score > 0.3 or auc < 0.7:
            frequency = "daily"
            reason = "high drift or low performance"
        elif drift_score > 0.1:
            frequency = "weekly"
            reason = "moderate drift detected"
        else:
            frequency = "monthly"
            reason = "model stable"

        next_run = datetime.now() + timedelta(
            days=1 if frequency == "daily" else 7 if frequency == "weekly" else 30
        )

        return {
            "frequency": frequency,
            "reason": reason,
            "next_run": next_run.isoformat(),
            "dataset_path": state.get("dataset_path", ""),
            "goal": state.get("goal", ""),
            "auto_retrain": drift_score > 0.2,
            "alert_on_drift": True,
            "alert_threshold": 0.2
        }

    def _record_current_run(self, state: dict) -> dict:
        """Record current pipeline run."""
        model_info = state.get("model_info", {})
        drift_report = state.get("drift_report", {})

        return {
            "run_id": state.get("run_id", ""),
            "timestamp": datetime.now().isoformat(),
            "goal": state.get("goal", ""),
            "dataset_path": state.get("dataset_path", ""),
            "model_name": model_info.get("model_name", ""),
            "auc": model_info.get("test_metrics", {}).get("auc_roc", 0),
            "f1": model_info.get("test_metrics", {}).get("f1_score", 0),
            "drift_score": drift_report.get("drift_score", 0),
            "retraining_triggered": state.get("retraining_results", {}).get("triggered", False),
            "pipeline_complete": state.get("pipeline_complete", False),
            "agents_run": len(state.get("planner_decision_log", []))
        }

    def run(self, state: dict) -> dict:
        """
        Pipeline Orchestration:
        1. Record current run
        2. Update run history
        3. Generate schedule
        4. Create run summary
        """

        self.log("Recording current pipeline run...")
        current_run = self._record_current_run(state)

        run_history = self._load_run_history()
        run_history.append(current_run)
        if len(run_history) > 100:
            run_history = run_history[-100:]
        self._save_run_history(run_history)

        self.log("Total runs in history: " + str(len(run_history)))

        # Generate schedule
        self.log("Generating pipeline schedule...")
        schedule = self._generate_schedule_config(state)
        self.log("Recommended frequency: " + schedule["frequency"])
        self.log("Reason: " + schedule["reason"])
        self.log("Next run: " + schedule["next_run"])

        # Trend analysis
        if len(run_history) >= 2:
            recent_aucs = [r["auc"] for r in run_history[-5:] if r["auc"]]
            if len(recent_aucs) >= 2:
                trend = "improving" if recent_aucs[-1] > recent_aucs[0] else "declining" if recent_aucs[-1] < recent_aucs[0] else "stable"
                self.log("Performance trend: " + trend)
            else:
                trend = "insufficient_data"
        else:
            trend = "first_run"

        orchestration_results = {
            "current_run": current_run,
            "total_runs": len(run_history),
            "schedule": schedule,
            "performance_trend": trend,
            "run_history_path": "outputs/run_history.json"
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        results_path = "outputs/" + run_id + "_orchestration.json"
        with open(results_path, "w") as f:
            json.dump(orchestration_results, f, indent=2)

        state["orchestration"] = orchestration_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] PipelineOrchestratorAgent: " +
            "run recorded, next_run=" + schedule["next_run"][:10] +
            " frequency=" + schedule["frequency"]
        )

        self.log("=" * 50)
        self.log("ORCHESTRATION COMPLETE")
        self.log("Run ID      : " + str(current_run["run_id"]))
        self.log("Total runs  : " + str(len(run_history)))
        self.log("Frequency   : " + schedule["frequency"])
        self.log("Next run    : " + schedule["next_run"][:10])
        self.log("Trend       : " + trend)
        self.log("=" * 50)

        return state