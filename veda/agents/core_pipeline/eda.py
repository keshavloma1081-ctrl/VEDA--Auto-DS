"""
VEDA — Autonomous Data Science System
agents/core_pipeline/eda.py — EDA Agent
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class EDAAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="EDAAgent",
            domain="ml",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_data(self, state: dict) -> pd.DataFrame:
        outputs_dir = "outputs"
        files = [f for f in os.listdir(outputs_dir) if f.endswith("_data.parquet")]
        if not files:
            return pd.read_csv(state.get("dataset_path", ""))
        latest = sorted(files)[-1]
        return pd.read_parquet(os.path.join(outputs_dir, latest))

    def _compute_stats(self, df: pd.DataFrame, target_col: str = None) -> dict:
        stats = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats["numeric_summary"] = {}
        for col in numeric_cols:
            stats["numeric_summary"][col] = {
                "mean": round(float(df[col].mean()), 4),
                "median": round(float(df[col].median()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "null_pct": round(float(df[col].isnull().mean() * 100), 2),
            }

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        stats["categorical_summary"] = {}
        for col in cat_cols:
            vc = df[col].value_counts()
            stats["categorical_summary"][col] = {
                "unique_count": int(df[col].nunique()),
                "null_pct": round(float(df[col].isnull().mean() * 100), 2),
                "top_values": vc.head(5).to_dict()
            }

        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            top_corr = upper.stack().sort_values(ascending=False).head(10)
            stats["top_correlations"] = {
                f"{i[0]} vs {i[1]}": round(float(v), 4)
                for i, v in top_corr.items()
            }

        stats["outlier_flags"] = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            if outliers > 0:
                stats["outlier_flags"][col] = int(outliers)

        stats["low_variance_cols"] = []
        for col in numeric_cols:
            if df[col].std() < 0.01:
                stats["low_variance_cols"].append(col)

        if target_col and target_col in df.columns:
            target = df[target_col]
            if target.nunique() <= 20:
                vc = target.value_counts()
                stats["target_distribution"] = vc.to_dict()
                if len(vc) >= 2:
                    ratio = vc.iloc[-1] / vc.iloc[0]
                    stats["imbalance_ratio"] = round(float(ratio), 4)
                    stats["has_imbalance"] = ratio < 0.2
            else:
                stats["target_distribution"] = {
                    "mean": round(float(target.mean()), 4),
                    "std": round(float(target.std()), 4)
                }
                stats["has_imbalance"] = False

        return stats

    def _generate_summary(self, goal: str, stats: dict, task_type: str) -> str:
        prompt = f"""You are VEDA, an autonomous data science system.

Goal: "{goal}"
Task type: {task_type}

EDA findings:
{json.dumps(stats, indent=2)[:3000]}

Write a concise 4-6 sentence plain-English summary covering:
1. What the data looks like overall
2. Key issues found (nulls, outliers, imbalance)
3. Which features look most important
4. Any data quality concerns

Be specific — mention actual column names and numbers."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are VEDA, an expert data scientist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()

    def run(self, state: dict) -> dict:

        goal = state.get("goal", "")
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        task_type = state.get("execution_plan", {}).get("task_type", "classification")

        self.log("Loading dataset for EDA...")
        df = self._load_data(state)
        self.log(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

        self.log("Computing EDA statistics...")
        stats = self._compute_stats(df, target_col)

        self.log("Generating plain-English summary with Groq...")
        summary = self._generate_summary(goal, stats, task_type)

        if state.get("data_profile"):
            state["data_profile"]["eda_summary"] = summary
            state["data_profile"]["has_imbalance"] = stats.get("has_imbalance", False)
        else:
            state["data_profile"] = {"eda_summary": summary}

        state["planner_decision_log"] = state.get("planner_decision_log", [])
        state["planner_decision_log"].append(
            f"[{datetime.now().isoformat()}] EDAAgent: analysis complete"
        )
        state["planner_decision_log"].append(
            f"[{datetime.now().isoformat()}] EDA Summary: {summary}"
        )

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        stats_path = f"outputs/{run_id}_eda_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        self.log(f"EDA stats saved to: {stats_path}")

        self.log("=" * 50)
        self.log("EDA SUMMARY:")
        self.log(summary)
        self.log("=" * 50)

        return state