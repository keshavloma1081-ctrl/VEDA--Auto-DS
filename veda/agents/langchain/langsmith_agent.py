"""
VEDA — Autonomous Data Science System
agents/langchain/langsmith_agent.py — LangSmith Agent

Observability and tracing for LangChain:
- Trace LLM calls
- Track token usage
- Monitor latency
- Log evaluation results
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class LangSmithAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="LangSmithAgent",
            domain="langchain",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        self.traces = []

    def _trace_llm_call(self, name: str, prompt: str,
                         response: str, latency_ms: float,
                         tokens: int) -> dict:
        """Record an LLM call trace."""
        trace = {
            "trace_id": name + "_" + datetime.now().strftime("%H%M%S%f")[:10],
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round(latency_ms, 2),
            "tokens_used": tokens,
            "prompt_preview": prompt[:100],
            "response_preview": response[:100],
            "status": "success"
        }
        self.traces.append(trace)
        return trace

    def _make_traced_call(self, name: str, prompt: str) -> tuple:
        """Make a traced LLM call."""
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            latency = (time.perf_counter() - start) * 1000
            text = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if hasattr(response, "usage") else 50
            trace = self._trace_llm_call(name, prompt, text, latency, tokens)
            return text, trace
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            trace = {
                "name": name,
                "latency_ms": round(latency, 2),
                "status": "error",
                "error": str(e)
            }
            self.traces.append(trace)
            return "", trace

    def _compute_trace_stats(self) -> dict:
        """Compute statistics from traces."""
        if not self.traces:
            return {}

        successful = [t for t in self.traces if t.get("status") == "success"]
        latencies = [t["latency_ms"] for t in successful if "latency_ms" in t]
        tokens = [t.get("tokens_used", 0) for t in successful]

        return {
            "total_calls": len(self.traces),
            "successful_calls": len(successful),
            "error_calls": len(self.traces) - len(successful),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
            "total_tokens": sum(tokens),
            "avg_tokens_per_call": round(sum(tokens) / max(len(successful), 1), 2)
        }

    def run(self, state: dict) -> dict:
        """
        LangSmith Agent:
        1. Make several traced LLM calls
        2. Track latency and token usage
        3. Compute trace statistics
        4. Generate observability report
        """

        goal = state.get("goal", "")
        model_name = state.get("model_info", {}).get("model_name", "LightGBM")

        self.log("Making traced LLM calls...")

        # Traced call 1 — pipeline summary
        text1, trace1 = self._make_traced_call(
            "pipeline_summary",
            "Summarize this ML pipeline in one sentence: goal=" + goal + " model=" + model_name
        )
        self.log("Call 1 latency: " + str(trace1.get("latency_ms", 0)) + "ms")

        # Traced call 2 — recommendation
        text2, trace2 = self._make_traced_call(
            "deployment_recommendation",
            "Should this model be deployed? AUC=" + str(
                state.get("model_info", {}).get("test_metrics", {}).get("auc_roc", 0)
            ) + " Answer yes or no with brief reason."
        )
        self.log("Call 2 latency: " + str(trace2.get("latency_ms", 0)) + "ms")

        # Traced call 3 — next steps
        text3, trace3 = self._make_traced_call(
            "next_steps",
            "What are the top 3 next steps after training a " + model_name + " model? Be brief."
        )
        self.log("Call 3 latency: " + str(trace3.get("latency_ms", 0)) + "ms")

        stats = self._compute_trace_stats()
        self.log("Trace stats: " + str(stats))

        langsmith_report = {
            "traces": self.traces,
            "stats": stats,
            "outputs": {
                "pipeline_summary": text1,
                "deployment_recommendation": text2,
                "next_steps": text3
            }
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_langsmith_traces.json"
        with open(path, "w") as f:
            json.dump(langsmith_report, f, indent=2)

        state["lc_traces"] = langsmith_report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] LangSmithAgent: " +
            str(stats.get("total_calls", 0)) + " calls, " +
            str(stats.get("total_tokens", 0)) + " tokens, " +
            "avg_latency=" + str(stats.get("avg_latency_ms", 0)) + "ms"
        )

        self.log("=" * 50)
        self.log("LANGSMITH COMPLETE")
        self.log("Total calls    : " + str(stats.get("total_calls", 0)))
        self.log("Total tokens   : " + str(stats.get("total_tokens", 0)))
        self.log("Avg latency    : " + str(stats.get("avg_latency_ms", 0)) + "ms")
        self.log("=" * 50)

        return state