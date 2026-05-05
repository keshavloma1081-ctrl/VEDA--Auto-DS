"""
VEDA — Autonomous Data Science System
agents/core_pipeline/planner.py — Master Planner Agent

The brain of VEDA. Receives a goal + dataset path,
analyses the data, classifies the task, and builds
a step-by-step execution plan for all other agents.
Uses Groq for fast, free LLM inference.
"""

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent
from veda.core.state import VEDAState, ExecutionPlan, ExecutionStep

load_dotenv()


class PlannerAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="MasterPlanner",
            domain="orchestration",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _sniff_data(self, dataset_path: str) -> dict:
        """Quick data sniff — reads schema without loading full dataset."""
        try:
            df = pd.read_csv(dataset_path, nrows=100)
            return {
                "columns": list(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "shape": df.shape,
                "null_counts": df.isnull().sum().to_dict(),
                "sample": df.head(3).to_string()
            }
        except Exception as e:
            self.log(f"Could not sniff data: {e}", level="WARN")
            return {}

    def _build_prompt(self, goal: str, data_info: dict) -> str:
        """Build the planning prompt."""
        return f"""You are VEDA, an autonomous data science system.

A user has given you this goal: "{goal}"

Here is a preview of their dataset:
- Columns: {data_info.get('columns', 'unknown')}
- Data types: {data_info.get('dtypes', 'unknown')}
- Shape: {data_info.get('shape', 'unknown')}
- Null counts: {data_info.get('null_counts', 'unknown')}
- Sample rows:
{data_info.get('sample', 'not available')}

Respond ONLY with a valid JSON object in this exact format, nothing else:
{{
    "task_type": "classification",
    "reasoning": "explain why in 2 sentences",
    "steps": ["ingest", "eda", "cleaning", "feature_engineering", "model_selection", "hyperparameter_tuning", "training", "evaluation", "explainability", "dashboard", "report"],
    "data_summary": "one paragraph about what you see in the data"
}}

task_type must be one of: classification, regression, timeseries, clustering, nlp"""

    def run(self, state: dict) -> dict:
        """
        Main planner logic:
        1. Sniff the dataset
        2. Ask Groq LLM to classify task + build plan
        3. Write plan to state
        """

        goal = state.get("goal", "")
        dataset_path = state.get("dataset_path", "")

        self.log(f"Goal: {goal}")
        self.log(f"Dataset: {dataset_path}")

        # Step 1 — sniff the data
        self.log("Sniffing dataset...")
        data_info = self._sniff_data(dataset_path)

        # Step 2 — ask Groq to plan
        self.log("Asking Groq LLM to build execution plan...")
        prompt = self._build_prompt(goal, data_info)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are VEDA, an autonomous data science system. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )

        # Step 3 — parse response
        raw = response.choices[0].message.content.strip()

        # Clean JSON if wrapped in markdown
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        plan_data = json.loads(raw)

        # Step 4 — build ExecutionPlan
        steps = []
        for i, agent_name in enumerate(plan_data["steps"]):
            steps.append(ExecutionStep(
                step_number=i + 1,
                agent_name=agent_name,
                description=f"Run {agent_name} agent",
                status="pending"
            ))

        execution_plan = ExecutionPlan(
            goal=goal,
            task_type=plan_data["task_type"],
            steps=steps
        )

        # Step 5 — write to state
        state["execution_plan"] = execution_plan.model_dump()
        state["planner_decision_log"] = state.get("planner_decision_log", [])
        state["planner_decision_log"].append(
            f"[{datetime.now().isoformat()}] Task type: {plan_data['task_type']}"
        )
        state["planner_decision_log"].append(
            f"[{datetime.now().isoformat()}] Reasoning: {plan_data['reasoning']}"
        )
        state["planner_decision_log"].append(
            f"[{datetime.now().isoformat()}] Data summary: {plan_data['data_summary']}"
        )

        self.log(f"Task type detected: {plan_data['task_type']}")
        self.log(f"Steps planned: {plan_data['steps']}")

        return state