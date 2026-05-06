"""
VEDA — Autonomous Data Science System
agents/langchain/lc_evaluator.py — LangChain Evaluator Agent

Evaluates LangChain chain outputs:
- Relevance scoring
- Faithfulness check
- Answer completeness
- Chain performance metrics
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class LCEvaluatorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="LCEvaluatorAgent",
            domain="langchain",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _score_relevance(self, question: str, answer: str) -> dict:
        """Score answer relevance to question."""
        prompt = """Score this Q&A pair on relevance (1-5).
Return JSON: {"score": 4, "reason": "brief reason"}

Question: """ + question[:200] + """
Answer: """ + answer[:200] + """

Return valid JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=80
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except:
            return {"score": 3, "reason": "Default score"}

    def _evaluate_chain_output(self, chain_name: str,
                                output: str, context: str) -> dict:
        """Evaluate a single chain output."""
        prompt = """Evaluate this LangChain output for quality.
Return JSON: {"faithfulness": 4, "completeness": 3, "clarity": 5, "overall": 4}
Scores 1-5.

Chain: """ + chain_name + """
Output: """ + str(output)[:300] + """
Context: """ + str(context)[:200] + """

Return valid JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except:
            return {"faithfulness": 3, "completeness": 3, "clarity": 3, "overall": 3}

    def run(self, state: dict) -> dict:
        """
        LangChain Evaluator:
        1. Load chain outputs from state
        2. Score relevance for QA outputs
        3. Evaluate chain quality
        4. Generate evaluation report
        """

        lc_chains = state.get("lc_chains", {})
        lc_traces = state.get("lc_traces", {})
        goal = state.get("goal", "")

        evaluation_results = {}

        # Evaluate chain outputs
        if lc_chains:
            self.log("Evaluating chain outputs...")

            if "qa" in lc_chains:
                qa_score = self._score_relevance(
                    "What was the best model and its performance?",
                    str(lc_chains["qa"])
                )
                evaluation_results["qa_relevance"] = qa_score
                self.log("QA relevance score: " + str(qa_score.get("score", 0)) + "/5")

            if "summary" in lc_chains:
                summary_eval = self._evaluate_chain_output(
                    "summarization_chain",
                    str(lc_chains["summary"]),
                    goal
                )
                evaluation_results["summary_quality"] = summary_eval
                self.log("Summary quality: " + str(summary_eval.get("overall", 0)) + "/5")

        # Evaluate traced outputs
        if lc_traces:
            self.log("Evaluating traced outputs...")
            outputs = lc_traces.get("outputs", {})

            if "pipeline_summary" in outputs:
                pipeline_eval = self._evaluate_chain_output(
                    "pipeline_summary",
                    str(outputs["pipeline_summary"]),
                    goal
                )
                evaluation_results["pipeline_summary_quality"] = pipeline_eval
                self.log("Pipeline summary quality: " + str(pipeline_eval.get("overall", 0)) + "/5")

        # Compute overall scores
        all_scores = []
        for eval_result in evaluation_results.values():
            if isinstance(eval_result, dict):
                score = eval_result.get("overall", eval_result.get("score", 0))
                if score:
                    all_scores.append(float(score))

        avg_score = round(sum(all_scores) / max(len(all_scores), 1), 2)

        # Get trace performance
        trace_stats = lc_traces.get("stats", {}) if lc_traces else {}

        eval_report = {
            "evaluations": evaluation_results,
            "avg_quality_score": avg_score,
            "total_evaluated": len(evaluation_results),
            "trace_performance": trace_stats,
            "overall_grade": "A" if avg_score >= 4.5 else "B" if avg_score >= 3.5 else "C"
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_lc_evaluation.json"
        with open(path, "w") as f:
            json.dump(eval_report, f, indent=2)

        state["lc_evaluation"] = eval_report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] LCEvaluatorAgent: " +
            "avg_score=" + str(avg_score) +
            " grade=" + eval_report["overall_grade"]
        )

        self.log("=" * 50)
        self.log("LC EVALUATOR COMPLETE")
        self.log("Evaluated  : " + str(len(evaluation_results)))
        self.log("Avg score  : " + str(avg_score) + "/5")
        self.log("Grade      : " + eval_report["overall_grade"])
        self.log("=" * 50)

        return state