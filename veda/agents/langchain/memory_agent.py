"""
VEDA — Autonomous Data Science System
agents/langchain/memory_agent.py — Memory Agent

Manages conversation and pipeline memory:
- Short-term memory (recent context)
- Long-term memory (persistent facts)
- Entity memory (key entities)
- Summary memory (compressed history)
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class MemoryAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="MemoryAgent",
            domain="langchain",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        self.short_term = []
        self.long_term = {}
        self.entity_memory = {}

    def _extract_entities(self, text: str) -> dict:
        """Extract key entities from text using Groq."""
        prompt = """Extract key entities from this text.
Return JSON: {"model": "...", "metric": "...", "dataset": "...", "task": "..."}
Return null for missing entities. Return valid JSON only.

Text: """ + str(text)[:300]

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
            return {}

    def _compress_memory(self, memories: list) -> str:
        """Compress multiple memories into a summary."""
        if not memories:
            return "No memories available"

        combined = " | ".join([str(m)[:100] for m in memories[-10:]])
        prompt = """Compress these memories into a 2-sentence summary:

Memories: """ + combined + """

Summary:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except:
            return "Pipeline ran " + str(len(memories)) + " steps successfully."

    def _update_short_term(self, new_info: str):
        """Update short-term memory (last 10 items)."""
        self.short_term.append({
            "timestamp": datetime.now().isoformat(),
            "content": new_info
        })
        if len(self.short_term) > 10:
            self.short_term = self.short_term[-10:]

    def _update_long_term(self, key: str, value):
        """Update long-term persistent memory."""
        self.long_term[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }

    def run(self, state: dict) -> dict:
        """
        Memory Management:
        1. Extract entities from pipeline state
        2. Update short-term memory
        3. Update long-term memory
        4. Compress memory history
        5. Save memory snapshot
        """

        goal = state.get("goal", "")
        model_info = state.get("model_info", {})
        decision_log = state.get("planner_decision_log", [])

        self.log("Extracting entities from pipeline context...")
        entity_text = goal + " " + str(model_info.get("model_name", "")) + " " + str(decision_log[-1] if decision_log else "")
        entities = self._extract_entities(entity_text)
        self.entity_memory.update({k: v for k, v in entities.items() if v})
        self.log("Entities: " + str(self.entity_memory))

        self.log("Updating short-term memory...")
        for log_entry in decision_log[-5:]:
            self._update_short_term(str(log_entry))

        self.log("Updating long-term memory...")
        self._update_long_term("goal", goal)
        self._update_long_term("best_model", model_info.get("model_name", ""))
        self._update_long_term("best_auc", model_info.get("test_metrics", {}).get("auc_roc", 0))
        self._update_long_term("run_id", state.get("run_id", ""))

        self.log("Compressing memory history...")
        memory_texts = [m["content"] for m in self.short_term]
        compressed = self._compress_memory(memory_texts)
        self.log("Compressed: " + compressed[:100])

        memory_snapshot = {
            "short_term": self.short_term,
            "long_term": self.long_term,
            "entity_memory": self.entity_memory,
            "compressed_summary": compressed
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_memory.json"
        with open(path, "w") as f:
            json.dump(memory_snapshot, f, indent=2)

        state["lc_memory"] = memory_snapshot
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] MemoryAgent: " +
            str(len(self.short_term)) + " short-term, " +
            str(len(self.long_term)) + " long-term memories"
        )

        self.log("=" * 50)
        self.log("MEMORY AGENT COMPLETE")
        self.log("Short-term : " + str(len(self.short_term)))
        self.log("Long-term  : " + str(len(self.long_term)))
        self.log("Entities   : " + str(self.entity_memory))
        self.log("=" * 50)

        return state