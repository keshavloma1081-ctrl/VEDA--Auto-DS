"""
VEDA — Autonomous Data Science System
agents/langchain/tool_agent.py — Tool Agent

LangChain tool use agent:
- Calculator tool
- Data lookup tool
- Model info tool
- Web search tool simulation
"""

import os
import json
import math
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class ToolAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="ToolAgent",
            domain="langchain",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        self.tool_calls = []

    def _calculator_tool(self, expression: str) -> dict:
        """Safe calculator tool."""
        try:
            allowed = set("0123456789+-*/().,% ")
            if all(c in allowed for c in expression):
                result = eval(expression)
                return {"tool": "calculator", "input": expression, "output": round(float(result), 6)}
            return {"tool": "calculator", "input": expression, "output": "Invalid expression"}
        except Exception as e:
            return {"tool": "calculator", "input": expression, "output": "Error: " + str(e)}

    def _data_lookup_tool(self, key: str, state: dict) -> dict:
        """Look up data from pipeline state."""
        lookup_map = {
            "model_name": state.get("model_info", {}).get("model_name", "N/A"),
            "auc": state.get("model_info", {}).get("test_metrics", {}).get("auc_roc", "N/A"),
            "f1": state.get("model_info", {}).get("test_metrics", {}).get("f1_score", "N/A"),
            "rows": state.get("data_profile", {}).get("row_count", "N/A") if state.get("data_profile") else "N/A",
            "features": len(state.get("feature_list", [])),
            "goal": state.get("goal", "N/A")
        }
        value = lookup_map.get(key.lower(), "Key not found")
        return {"tool": "data_lookup", "input": key, "output": value}

    def _model_info_tool(self, state: dict) -> dict:
        """Get model information."""
        model_info = state.get("model_info", {})
        return {
            "tool": "model_info",
            "input": "model_info",
            "output": {
                "name": model_info.get("model_name", "N/A"),
                "path": model_info.get("model_path", "N/A"),
                "metrics": model_info.get("test_metrics", {})
            }
        }

    def _decide_tools(self, goal: str, state: dict) -> list:
        """Use LLM to decide which tools to call."""
        prompt = """You are a tool-use agent. Decide which tools to call.

Available tools: calculator, data_lookup, model_info
Goal: """ + goal + """

Return JSON list of tool calls:
[
    {"tool": "data_lookup", "input": "model_name"},
    {"tool": "data_lookup", "input": "auc"},
    {"tool": "calculator", "input": "0.97 * 100"}
]

Return valid JSON array only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except:
            return [
                {"tool": "data_lookup", "input": "model_name"},
                {"tool": "data_lookup", "input": "auc"},
                {"tool": "model_info", "input": "model_info"}
            ]

    def run(self, state: dict) -> dict:
        """
        Tool Agent:
        1. Decide which tools to call
        2. Execute tool calls
        3. Aggregate results
        """

        goal = state.get("goal", "analyze the ML pipeline results")
        self.log("Deciding which tools to use...")
        tool_calls = self._decide_tools(goal, state)
        self.log("Tools selected: " + str([t["tool"] for t in tool_calls]))

        results = []
        for call in tool_calls[:5]:
            tool = call.get("tool", "")
            inp = call.get("input", "")

            if tool == "calculator":
                result = self._calculator_tool(str(inp))
            elif tool == "data_lookup":
                result = self._data_lookup_tool(str(inp), state)
            elif tool == "model_info":
                result = self._model_info_tool(state)
            else:
                result = {"tool": tool, "input": inp, "output": "Unknown tool"}

            results.append(result)
            self.log("Tool: " + tool + " -> " + str(result["output"])[:50])

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_tool_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        state["lc_tool_results"] = results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ToolAgent: " +
            str(len(results)) + " tool calls executed"
        )

        self.log("=" * 50)
        self.log("TOOL AGENT COMPLETE")
        self.log("Tool calls: " + str(len(results)))
        self.log("=" * 50)

        return state