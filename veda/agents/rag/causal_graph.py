"""
VEDA — Autonomous Data Science System
agents/rag/causal_graph.py — Causal Graph Agent

Builds causal graphs from data:
- Correlation-based causal discovery
- DAG construction
- Causal path analysis
- LLM-powered causal reasoning
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class CausalGraphAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="CausalGraphAgent",
            domain="causal",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _compute_correlations(self, df: pd.DataFrame,
                               target_col: str) -> dict:
        """Compute correlations between all numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        correlations = {}
        if target_col and target_col in corr_matrix.columns:
            target_corr = corr_matrix[target_col].drop(target_col)
            correlations["with_target"] = {
                col: round(float(val), 4)
                for col, val in target_corr.items()
                if abs(val) > 0.1
            }

        top_pairs = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.3 and cols[i] != target_col and cols[j] != target_col:
                    top_pairs.append({
                        "feature1": cols[i],
                        "feature2": cols[j],
                        "correlation": round(float(corr_val), 4)
                    })

        top_pairs.sort(key=lambda x: x["correlation"], reverse=True)
        correlations["feature_pairs"] = top_pairs[:10]

        return correlations

    def _build_dag(self, correlations: dict,
                   target_col: str) -> dict:
        """Build a simple DAG from correlations."""
        nodes = set()
        edges = []

        with_target = correlations.get("with_target", {})
        for feature, corr in with_target.items():
            nodes.add(feature)
            nodes.add(target_col)
            direction = "causes" if corr > 0 else "inversely_causes"
            edges.append({
                "from": feature,
                "to": target_col,
                "weight": abs(corr),
                "direction": direction
            })

        for pair in correlations.get("feature_pairs", [])[:5]:
            nodes.add(pair["feature1"])
            nodes.add(pair["feature2"])
            edges.append({
                "from": pair["feature1"],
                "to": pair["feature2"],
                "weight": pair["correlation"],
                "direction": "correlates"
            })

        return {
            "nodes": list(nodes),
            "edges": edges,
            "target": target_col
        }

    def _llm_causal_reasoning(self, dag: dict, goal: str) -> str:
        """Use LLM for causal reasoning."""
        top_edges = sorted(dag["edges"], key=lambda x: x["weight"], reverse=True)[:5]

        prompt = """You are a causal inference expert. Analyze this causal graph.

Goal: """ + goal + """

Top causal relationships (by strength):
""" + json.dumps(top_edges, indent=2) + """

Write 3 sentences explaining:
1. The main causal drivers
2. Any potential confounders
3. What interventions might work

Be specific and use feature names."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Causal analysis complete. See DAG for relationships."

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        goal = state.get("goal", "")

        self.log("Loading features for causal analysis...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        self.log("Computing correlations...")
        correlations = self._compute_correlations(df, target_col)

        with_target = correlations.get("with_target", {})
        self.log("Features correlated with target: " + str(len(with_target)))

        self.log("Building causal DAG...")
        dag = self._build_dag(correlations, target_col)
        self.log("DAG: " + str(len(dag["nodes"])) + " nodes, " +
                str(len(dag["edges"])) + " edges")

        self.log("Running LLM causal reasoning...")
        reasoning = self._llm_causal_reasoning(dag, goal)
        self.log("Reasoning: " + reasoning[:100])

        causal_results = {
            "correlations": correlations,
            "dag": dag,
            "causal_reasoning": reasoning,
            "top_causal_features": list(with_target.keys())[:5]
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_causal_graph.json"
        with open(path, "w") as f:
            json.dump(causal_results, f, indent=2)

        state["causal_graph"] = causal_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] CausalGraphAgent: " +
            str(len(dag["nodes"])) + " nodes, " +
            str(len(dag["edges"])) + " edges"
        )

        self.log("=" * 50)
        self.log("CAUSAL GRAPH COMPLETE")
        self.log("Nodes    : " + str(len(dag["nodes"])))
        self.log("Edges    : " + str(len(dag["edges"])))
        self.log("Top causes: " + str(list(with_target.keys())[:3]))
        self.log("=" * 50)

        return state