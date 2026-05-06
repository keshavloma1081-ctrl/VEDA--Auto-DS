"""
VEDA — Autonomous Data Science System
agents/llm/llm_chain.py — LLM Chain Agent

Builds LangChain prompt chains for:
- Data analysis narration
- Insight generation
- Question answering over data
"""

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class LLMChainAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="LLMChainAgent",
            domain="llm",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_context(self, state: dict) -> dict:
        """Build context from pipeline state."""
        data_profile = state.get("data_profile", {})
        model_info = state.get("model_info", {})
        explainability = state.get("explainability", {})

        return {
            "goal": state.get("goal", ""),
            "rows": data_profile.get("row_count", 0),
            "columns": data_profile.get("col_count", 0),
            "target": data_profile.get("target_column", ""),
            "model_name": model_info.get("model_name", ""),
            "auc": model_info.get("test_metrics", {}).get("auc_roc", 0),
            "f1": model_info.get("test_metrics", {}).get("f1_score", 0),
            "top_features": explainability.get("top_features", []),
            "eda_summary": data_profile.get("eda_summary", ""),
            "cleaning_diff": state.get("cleaning_diff", [])
        }

    def _run_chain(self, chain_name: str, prompt: str) -> str:
        """Run a single LLM chain."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are VEDA, an expert data scientist. Be concise, specific, and actionable."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.log("Chain " + chain_name + " failed: " + str(e), level="WARN")
            return "Chain execution failed."

    def _chain_1_data_story(self, ctx: dict) -> str:
        """Chain 1 — Generate data story."""
        prompt = """You are a data scientist presenting findings to stakeholders.

Project: """ + ctx["goal"] + """
Dataset: """ + str(ctx["rows"]) + """ rows, """ + str(ctx["columns"]) + """ columns
Target: """ + str(ctx["target"]) + """
EDA Summary: """ + str(ctx["eda_summary"])[:300] + """

Write a 3-sentence data story that explains:
1. What the data represents
2. Key patterns found
3. Why this matters for the business goal"""
        return self._run_chain("data_story", prompt)

    def _chain_2_model_insights(self, ctx: dict) -> str:
        """Chain 2 — Generate model insights."""
        prompt = """You are explaining ML model results to a business audience.

Model: """ + str(ctx["model_name"]) + """
AUC: """ + str(ctx["auc"]) + """
F1: """ + str(ctx["f1"]) + """
Top features: """ + str(ctx["top_features"][:5]) + """

Write 3 bullet points explaining:
- What the model learned
- Which features matter most and why
- What this means for the business

Use plain English, no technical jargon."""
        return self._run_chain("model_insights", prompt)

    def _chain_3_recommendations(self, ctx: dict) -> str:
        """Chain 3 — Generate actionable recommendations."""
        prompt = """Based on this ML analysis, generate actionable recommendations.

Goal: """ + ctx["goal"] + """
Model performance: AUC=""" + str(ctx["auc"]) + """ F1=""" + str(ctx["f1"]) + """
Key drivers: """ + str(ctx["top_features"][:5]) + """
Data issues fixed: """ + str(ctx["cleaning_diff"][:3]) + """

Provide 3 specific, actionable recommendations the business should take.
Format as numbered list."""
        return self._run_chain("recommendations", prompt)

    def _chain_4_risk_assessment(self, ctx: dict) -> str:
        """Chain 4 — Risk assessment chain."""
        prompt = """Assess the risks and limitations of this ML model.

Goal: """ + ctx["goal"] + """
Model: """ + str(ctx["model_name"]) + """
AUC: """ + str(ctx["auc"]) + """
Training rows: """ + str(ctx["rows"]) + """

Identify 3 key risks or limitations:
1. Data risks
2. Model risks  
3. Deployment risks

Be specific and suggest mitigations."""
        return self._run_chain("risk_assessment", prompt)

    def run(self, state: dict) -> dict:
        """
        LLM Chain Pipeline:
        1. Build context from state
        2. Run 4 analysis chains
        3. Save all outputs
        """

        self.log("Building context from pipeline state...")
        ctx = self._load_context(state)
        self.log("Context built — goal: " + ctx["goal"][:50])

        chains = {}

        # Chain 1 — Data Story
        self.log("Running Chain 1: Data Story...")
        chains["data_story"] = self._chain_1_data_story(ctx)
        self.log("Chain 1 done")

        # Chain 2 — Model Insights
        self.log("Running Chain 2: Model Insights...")
        chains["model_insights"] = self._chain_2_model_insights(ctx)
        self.log("Chain 2 done")

        # Chain 3 — Recommendations
        self.log("Running Chain 3: Recommendations...")
        chains["recommendations"] = self._chain_3_recommendations(ctx)
        self.log("Chain 3 done")

        # Chain 4 — Risk Assessment
        self.log("Running Chain 4: Risk Assessment...")
        chains["risk_assessment"] = self._chain_4_risk_assessment(ctx)
        self.log("Chain 4 done")

        # Save results
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        results_path = "outputs/" + run_id + "_llm_chains.json"
        with open(results_path, "w") as f:
            json.dump(chains, f, indent=2)

        state["llm_chains"] = chains
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] LLMChainAgent: 4 chains completed"
        )

        self.log("=" * 50)
        self.log("LLM CHAINS COMPLETE")
        self.log("Data Story     : " + chains["data_story"][:100] + "...")
        self.log("Recommendations: " + chains["recommendations"][:100] + "...")
        self.log("=" * 50)

        return state