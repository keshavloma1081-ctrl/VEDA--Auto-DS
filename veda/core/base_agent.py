"""
VEDA — Autonomous Data Science System
core/base_agent.py — Base class for all 128 VEDA agents

Every agent inherits from this class. It handles:
- Reading inputs from VEDAState
- Writing outputs back to VEDAState
- Registering with the health monitoring system
- Error handling and logging
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
import traceback


class BaseAgent(ABC):
    """
    Parent class for every VEDA agent.
    Inherit from this and implement the run() method.
    """

    def __init__(self, name: str, domain: str, version: str = "1.0.0"):
        self.name = name
        self.domain = domain
        self.version = version
        self.created_at = datetime.now().isoformat()
        self.error_count = 0
        self.last_run: Optional[str] = None

    # ── Logging ──────────────────────────────────────────────
    def log(self, message: str, level: str = "INFO"):
        """Simple structured logger."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] [{self.name}] {message}")

    def log_start(self):
        self.last_run = datetime.now().isoformat()
        self.log(f"Starting — version {self.version}")

    def log_done(self):
        self.log("Done ✓")

    def log_error(self, error: Exception):
        self.error_count += 1
        self.log(f"Error: {str(error)}", level="ERROR")
        self.log(traceback.format_exc(), level="ERROR")

    # ── Main execution wrapper ───────────────────────────────
    def execute(self, state) -> dict:
        """
        Called by LangGraph for every agent node.
        Wraps run() with health tracking and error handling.
        """
        self.log_start()

        # ── Convert Pydantic state to dict if needed ─────────
        if not isinstance(state, dict):
            state = dict(state)

        # ── Initialise health registry if missing ─────────────
        if "agent_health_registry" not in state:
            state["agent_health_registry"] = {}

        # ── Register this agent in health registry ────────────
        if self.name not in state["agent_health_registry"]:
            state["agent_health_registry"][self.name] = {
                "agent_name": self.name,
                "health_score": 100.0,
                "status": "healthy",
                "error_count": 0,
                "last_ping": datetime.now().isoformat(),
                "replacing": False
            }

        try:
            # ── Run the agent ─────────────────────────────────
            state = self.run(state)

            # ── Update health on success ──────────────────────
            health = state["agent_health_registry"][self.name]
            health["health_score"] = min(100.0, health["health_score"] + 5.0)
            health["status"] = "healthy"
            health["last_ping"] = datetime.now().isoformat()

            self.log_done()

        except Exception as e:
            self.log_error(e)

            # ── Update health on failure ──────────────────────
            health = state["agent_health_registry"][self.name]
            health["error_count"] += 1
            health["health_score"] = max(0.0, health["health_score"] - 25.0)

            if health["health_score"] > 70:
                health["status"] = "healthy"
            elif health["health_score"] > 35:
                health["status"] = "degraded"
            else:
                health["status"] = "critical"

            # ── Log failure to state ──────────────────────────
            if "planner_decision_log" not in state:
                state["planner_decision_log"] = []

            state["planner_decision_log"].append(
                f"[{datetime.now().isoformat()}] {self.name} FAILED: {str(e)}"
            )

            # ── Flag for replacement if critical ──────────────
            if health["status"] == "critical":
                self.log(
                    f"Health critical ({health['health_score']}%) — flagging for replacement",
                    level="WARN"
                )
                state["human_review_required"] = True

        return state

    # ── Abstract method ──────────────────────────────────────
    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Implement this in every agent subclass.
        Read from state → do work → write back to state → return state.
        """
        pass

    # ── String representation ────────────────────────────────
    def __repr__(self):
        return f"<VEDAAgent name={self.name} domain={self.domain} v{self.version}>"