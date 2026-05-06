"""
VEDA — Autonomous Data Science System
agents/automl/automl_report.py — AutoML Report Agent

Generates comprehensive AutoML report:
- Model comparison
- Feature selection summary
- Compression recommendations
- Deployment readiness
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class AutoMLReportAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="AutoMLReportAgent",
            domain="automl",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _generate_summary(self, state: dict) -> str:
        automl = state.get("automl_results", {})
        hyperopt = state.get("hyperopt_results", {})
        features = state.get("feature_selection", {})
        compression = state.get("compression_results", {})
        goal = state.get("goal", "")

        prompt = """You are an AutoML expert. Summarize these results.

Goal: """ + goal + """

AutoML best model: """ + str(automl.get("best_estimator", "N/A")) + """
AutoML AUC: """ + str(automl.get("auc", "N/A")) + """

Hyperopt best AUC: """ + str(hyperopt.get("best_auc", "N/A")) + """
Best params: """ + str(hyperopt.get("best_params", {})) + """

Feature selection: """ + str(features.get("original_features", 0)) + """ -> """ + str(features.get("final_selected", 0)) + """ features

Model size: """ + str(compression.get("original_model_size", {}).get("size_kb", "N/A")) + """ KB

Write 4 sentences:
1. Best model found and performance
2. Feature engineering impact
3. Optimization improvements
4. Deployment recommendation"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except:
            return "AutoML search complete. Best model: " + str(automl.get("best_estimator", "N/A"))

    def _deployment_readiness(self, state: dict) -> dict:
        """Check deployment readiness."""
        automl = state.get("automl_results", {})
        auc = automl.get("auc", 0)
        checks = {
            "performance_threshold": bool(auc >= 0.7),
            "model_exists": bool(automl.get("best_estimator")),
            "features_selected": bool(state.get("selected_features")),
            "hyperparams_optimized": bool(state.get("hyperopt_results")),
            "compression_analyzed": bool(state.get("compression_results"))
        }
        passed = sum(checks.values())
        total = len(checks)
        return {
            "checks": checks,
            "passed": passed,
            "total": total,
            "ready": passed >= 3,
            "score": round(passed / total * 100, 1)
        }

    def run(self, state: dict) -> dict:
        self.log("Generating AutoML report...")
        summary = self._generate_summary(state)

        self.log("Checking deployment readiness...")
        readiness = self._deployment_readiness(state)
        self.log("Deployment readiness: " + str(readiness["score"]) + "%")

        automl = state.get("automl_results", {})
        hyperopt = state.get("hyperopt_results", {})
        features = state.get("feature_selection", {})
        compression = state.get("compression_results", {})

        report = {
            "summary": summary,
            "deployment_readiness": readiness,
            "best_model": automl.get("best_estimator", "N/A"),
            "best_auc": automl.get("auc", "N/A"),
            "optimized_auc": hyperopt.get("best_auc", "N/A"),
            "features_reduced": str(features.get("original_features", 0)) +
                               " -> " + str(features.get("final_selected", 0)),
            "model_size_kb": compression.get("original_model_size", {}).get("size_kb", "N/A"),
            "recommendations": [
                "Deploy " + str(automl.get("best_estimator", "best model")) + " with optimized hyperparameters",
                "Use " + str(features.get("final_selected", "selected")) + " features for inference",
                "Apply INT8 quantization for 4x size reduction"
            ]
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_automl_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        state["automl_report"] = report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] AutoMLReportAgent: " +
            "readiness=" + str(readiness["score"]) + "% " +
            "best=" + str(automl.get("best_estimator", "N/A"))
        )

        self.log("=" * 50)
        self.log("AUTOML REPORT COMPLETE")
        self.log("Best model  : " + str(automl.get("best_estimator", "N/A")))
        self.log("Readiness   : " + str(readiness["score"]) + "%")
        self.log("Summary     : " + summary[:100] + "...")
        self.log("=" * 50)

        return state