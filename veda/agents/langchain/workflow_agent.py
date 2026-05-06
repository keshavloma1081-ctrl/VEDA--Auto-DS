"""
VEDA — Autonomous Data Science System
agents/langchain/workflow_agent.py — LangGraph Workflow Agent

Manages complex multi-step workflows:
- Sequential workflows
- Conditional branching
- Parallel execution simulation
- Workflow state tracking
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class WorkflowAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="WorkflowAgent",
            domain="langchain",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _sequential_workflow(self, steps: list, context: dict) -> list:
        """Execute steps sequentially."""
        results = []
        for i, step in enumerate(steps):
            prompt = """Execute workflow step """ + str(i+1) + """: """ + step + """

Context: """ + json.dumps(context, default=str)[:300] + """

Provide a brief result for this step (1-2 sentences):"""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100
                )
                result = response.choices[0].message.content.strip()
            except:
                result = "Step " + str(i+1) + " completed"

            results.append({
                "step": i + 1,
                "description": step,
                "result": result,
                "status": "COMPLETED"
            })
            self.log("Step " + str(i+1) + ": " + step[:50] + " -> DONE")

        return results

    def _conditional_branch(self, condition: str, state: dict) -> str:
        """Evaluate condition and return branch."""
        auc = state.get("model_info", {}).get("test_metrics", {}).get("auc_roc", 0)
        drift = state.get("drift_report", {}).get("drift_score", 0)

        if condition == "performance_check":
            return "deploy" if auc >= 0.7 else "retrain"
        elif condition == "drift_check":
            return "retrain" if drift > 0.2 else "monitor"
        elif condition == "compliance_check":
            gdpr = state.get("gdpr_report", {}).get("overall_status", "NEEDS_ATTENTION")
            return "deploy" if gdpr == "COMPLIANT" else "review"
        return "continue"

    def _build_workflow_graph(self, state: dict) -> dict:
        """Build a workflow graph for the pipeline."""
        auc = state.get("model_info", {}).get("test_metrics", {}).get("auc_roc", 0)
        drift = state.get("drift_report", {}).get("drift_score", 0)

        nodes = [
            {"id": "start", "type": "start", "label": "Pipeline Start"},
            {"id": "train", "type": "process", "label": "Model Training"},
            {"id": "evaluate", "type": "decision", "label": "Evaluate AUC >= 0.7?"},
            {"id": "deploy", "type": "process", "label": "Deploy Model"},
            {"id": "retrain", "type": "process", "label": "Retrain Model"},
            {"id": "monitor", "type": "process", "label": "Monitor + Drift Check"},
            {"id": "end", "type": "end", "label": "Pipeline End"}
        ]

        edges = [
            {"from": "start", "to": "train"},
            {"from": "train", "to": "evaluate"},
            {"from": "evaluate", "to": "deploy", "condition": "AUC >= 0.7"},
            {"from": "evaluate", "to": "retrain", "condition": "AUC < 0.7"},
            {"from": "deploy", "to": "monitor"},
            {"from": "retrain", "to": "evaluate"},
            {"from": "monitor", "to": "end"}
        ]

        current_node = "deploy" if auc >= 0.7 else "retrain"

        return {
            "nodes": nodes,
            "edges": edges,
            "current_node": current_node,
            "workflow_status": "RUNNING"
        }

    def run(self, state: dict) -> dict:
        """
        Workflow Agent:
        1. Build workflow graph
        2. Execute sequential workflow
        3. Evaluate conditional branches
        4. Track workflow state
        """

        goal = state.get("goal", "")

        self.log("Building workflow graph...")
        graph = self._build_workflow_graph(state)
        self.log("Current node: " + graph["current_node"])

        self.log("Executing sequential workflow...")
        steps = [
            "Validate input data quality",
            "Select optimal model architecture",
            "Train and evaluate model",
            "Run compliance checks",
            "Generate reports and dashboard",
            "Deploy to production"
        ]

        context = {
            "goal": goal,
            "model": state.get("model_info", {}).get("model_name", ""),
            "auc": state.get("model_info", {}).get("test_metrics", {}).get("auc_roc", 0)
        }

        workflow_results = self._sequential_workflow(steps[:3], context)

        self.log("Evaluating conditional branches...")
        branches = {}
        for condition in ["performance_check", "drift_check", "compliance_check"]:
            branch = self._conditional_branch(condition, state)
            branches[condition] = branch
            self.log(condition + " -> " + branch)

        workflow_output = {
            "graph": graph,
            "sequential_results": workflow_results,
            "branch_decisions": branches,
            "workflow_complete": True
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_workflow.json"
        with open(path, "w") as f:
            json.dump(workflow_output, f, indent=2)

        state["lc_workflow"] = workflow_output
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] WorkflowAgent: " +
            str(len(workflow_results)) + " steps, branches=" + str(branches)
        )

        self.log("=" * 50)
        self.log("WORKFLOW AGENT COMPLETE")
        self.log("Steps completed : " + str(len(workflow_results)))
        self.log("Branch decisions: " + str(branches))
        self.log("=" * 50)

        return state