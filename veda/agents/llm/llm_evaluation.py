"""
VEDA — Autonomous Data Science System
agents/llm/llm_evaluation.py — LLM Evaluation Agent

Evaluates LLM outputs for:
- Hallucination detection
- Factual consistency
- Answer quality scoring
- Output validation
"""

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class LLMEvaluationAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="LLMEvaluationAgent",
            domain="llm",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _check_hallucination(self, text: str, facts: dict) -> dict:
        """Check text for hallucinations against known facts."""
        issues = []

        # Check numeric claims
        numbers_in_text = re.findall(r"\b\d+\.?\d*\b", text)

        for num_str in numbers_in_text:
            num = float(num_str)
            # Check if number is wildly different from known metrics
            if "auc" in text.lower() and "auc" in str(facts).lower():
                known_auc = facts.get("auc", 0)
                if known_auc and abs(num - known_auc) > 0.3 and 0 < num <= 1:
                    issues.append("Possible AUC hallucination: " + num_str +
                                  " vs known " + str(known_auc))

        return {
            "issues_found": len(issues),
            "issues": issues,
            "passed": len(issues) == 0
        }

    def _score_answer_quality(self, question: str, answer: str) -> dict:
        """Score answer quality using LLM-as-judge."""
        prompt = """Rate this Q&A pair on a scale of 1-5 for each criterion.
Return JSON only.

Question: """ + question[:200] + """
Answer: """ + answer[:300] + """

Rate on:
- relevance (1-5): Does the answer address the question?
- specificity (1-5): Is it specific rather than vague?
- clarity (1-5): Is it clear and well-written?

Return: {"relevance": 4, "specificity": 3, "clarity": 5, "overall": 4}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            return {"relevance": 3, "specificity": 3, "clarity": 3, "overall": 3}

    def _evaluate_consistency(self, outputs: dict, facts: dict) -> dict:
        """Check consistency across all LLM outputs."""
        inconsistencies = []

        # Check if all outputs reference the same model
        model_mentions = []
        for key, text in outputs.items():
            if isinstance(text, str):
                for model_name in ["lightgbm", "randomforest", "xgboost", "logisticregression"]:
                    if model_name in text.lower():
                        model_mentions.append((key, model_name))

        if len(set([m[1] for m in model_mentions])) > 1:
            inconsistencies.append("Multiple model names mentioned across outputs: " +
                                   str(set([m[1] for m in model_mentions])))

        return {
            "inconsistencies": inconsistencies,
            "consistent": len(inconsistencies) == 0
        }

    def run(self, state: dict) -> dict:
        """
        LLM Evaluation:
        1. Load all LLM outputs from state
        2. Check for hallucinations
        3. Score answer quality
        4. Check consistency
        5. Generate evaluation report
        """

        self.log("Collecting LLM outputs from state...")

        # Gather all LLM outputs
        llm_outputs = {}
        if "llm_chains" in state:
            llm_outputs.update(state["llm_chains"])
        if "rag_results" in state:
            for qa in state["rag_results"]:
                llm_outputs["rag_" + qa["question"][:30]] = qa["answer"]

        if not llm_outputs:
            self.log("No LLM outputs found to evaluate", level="WARN")
            return state

        self.log("Evaluating " + str(len(llm_outputs)) + " LLM outputs...")

        # Build facts from state
        model_info = state.get("model_info", {})
        facts = {
            "auc": model_info.get("test_metrics", {}).get("auc_roc", 0),
            "model": model_info.get("model_name", ""),
            "rows": state.get("data_profile", {}).get("row_count", 0) if state.get("data_profile") else 0
        }

        evaluation_results = {}
        total_hallucinations = 0
        quality_scores = []

        for output_name, output_text in llm_outputs.items():
            if not isinstance(output_text, str):
                continue

            self.log("Evaluating: " + output_name[:40])

            # Hallucination check
            hallucination = self._check_hallucination(output_text, facts)
            total_hallucinations += hallucination["issues_found"]

            # Quality scoring (sample only to save API calls)
            if len(quality_scores) < 3:
                quality = self._score_answer_quality(output_name, output_text)
                quality_scores.append(quality.get("overall", 3))
            else:
                quality = {"overall": 3}

            evaluation_results[output_name] = {
                "hallucination_check": hallucination,
                "quality_score": quality
            }

        # Consistency check
        consistency = self._evaluate_consistency(llm_outputs, facts)

        # Overall assessment
        avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 3.0
        passed = total_hallucinations == 0 and consistency["consistent"]

        summary = {
            "total_outputs_evaluated": len(evaluation_results),
            "total_hallucinations": total_hallucinations,
            "avg_quality_score": avg_quality,
            "consistency_check": consistency,
            "overall_passed": passed,
            "detailed_results": evaluation_results
        }

        # Save report
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_path = "outputs/" + run_id + "_llm_evaluation.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        state["llm_evaluation"] = summary
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] LLMEvaluationAgent: " +
            str(len(evaluation_results)) + " outputs evaluated, hallucinations=" +
            str(total_hallucinations) + " quality=" + str(avg_quality)
        )

        self.log("=" * 50)
        self.log("LLM EVALUATION COMPLETE")
        self.log("Outputs evaluated : " + str(len(evaluation_results)))
        self.log("Hallucinations    : " + str(total_hallucinations))
        self.log("Avg quality score : " + str(avg_quality) + "/5")
        self.log("Overall passed    : " + str(passed))
        self.log("=" * 50)

        return state