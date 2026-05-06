"""
VEDA — Autonomous Data Science System
monitors/health_monitor.py — Agent Health Monitor

Watches all active agents continuously.
Detects degraded or failed agents.
Triggers replacement from idle pool.
"""

import time
import threading
from datetime import datetime
from veda.core.base_agent import BaseAgent


class AgentHealthMonitor:
    """
    Runs as a background thread.
    Pings every registered agent every N seconds.
    Fires replacement when health drops below threshold.
    """

    def __init__(self, check_interval: int = 5, critical_threshold: float = 35.0):
        self.check_interval = check_interval
        self.critical_threshold = critical_threshold
        self.registered_agents = {}
        self.idle_pool = {}
        self.replacement_count = 0
        self.running = False
        self._thread = None
        self.event_log = []

    def register_agent(self, agent: BaseAgent):
        """Register an active agent for monitoring."""
        self.registered_agents[agent.name] = agent
        self._log("REGISTERED active agent: " + agent.name)

    def register_idle(self, agent: BaseAgent, covers: str):
        """Register an idle agent as backup for a domain."""
        self.idle_pool[covers] = agent
        self._log("REGISTERED idle agent: " + agent.name + " covers=" + covers)

    def _log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = "[" + timestamp + "] [" + level + "] [HealthMonitor] " + message
        print(entry)
        self.event_log.append(entry)

    def _ping_agent(self, agent: BaseAgent) -> float:
        """
        Ping an agent and return its current health score.
        In production this would call agent.health_check().
        For now we read from the agent's error_count.
        """
        try:
            # Simulate health based on error count
            health = max(0.0, 100.0 - (agent.error_count * 25.0))
            return health
        except Exception:
            return 0.0

    def _find_replacement(self, agent_name: str, domain: str) -> BaseAgent:
        """Find the best idle agent to replace a degraded agent."""
        # Try exact domain match first
        if domain in self.idle_pool:
            return self.idle_pool[domain]
        # Fall back to any available idle agent
        if self.idle_pool:
            return list(self.idle_pool.values())[0]
        return None

    def _replace_agent(self, agent: BaseAgent, state: dict = None):
        """Replace a degraded agent with an idle one."""
        replacement = self._find_replacement(agent.name, agent.domain)

        if not replacement:
            self._log(
                "No idle agent available to replace " + agent.name,
                level="WARN"
            )
            return

        self._log(
            "REPLACING " + agent.name + " with " + replacement.name,
            level="WARN"
        )

        # Reset the replacement agent's error count
        replacement.error_count = 0

        # Swap in registered agents
        self.registered_agents[agent.name] = replacement

        self.replacement_count += 1
        self._log(
            "REPLACEMENT COMPLETE — " + replacement.name +
            " now serving as " + agent.name +
            " (total replacements: " + str(self.replacement_count) + ")"
        )

    def _check_all_agents(self, state: dict = None):
        """Run one health check cycle across all registered agents."""
        for name, agent in list(self.registered_agents.items()):
            health = self._ping_agent(agent)

            if health > 70:
                status = "healthy"
            elif health > 35:
                status = "degraded"
                self._log(
                    name + " DEGRADED (health=" + str(round(health, 1)) + "%)",
                    level="WARN"
                )
            else:
                status = "critical"
                self._log(
                    name + " CRITICAL (health=" + str(round(health, 1)) + "%) — triggering replacement",
                    level="ERROR"
                )
                self._replace_agent(agent, state)

    def _monitor_loop(self, state: dict = None):
        """Background monitoring loop."""
        self._log("Health monitoring started — checking every " + str(self.check_interval) + "s")
        while self.running:
            self._check_all_agents(state)
            time.sleep(self.check_interval)
        self._log("Health monitoring stopped")

    def start(self, state: dict = None):
        """Start background health monitoring."""
        self.running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(state,),
            daemon=True
        )
        self._thread.start()
        self._log("Monitor thread started")

    def stop(self):
        """Stop background health monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)
        self._log("Monitor stopped")

    def get_status_report(self) -> dict:
        """Return current health status of all agents."""
        report = {}
        for name, agent in self.registered_agents.items():
            health = self._ping_agent(agent)
            report[name] = {
                "health_score": round(health, 1),
                "error_count": agent.error_count,
                "status": "healthy" if health > 70 else "degraded" if health > 35 else "critical",
                "domain": agent.domain
            }
        return report

    def print_status(self):
        """Print a formatted health status table."""
        print("\n" + "="*55)
        print("  VEDA Agent Health Status")
        print("="*55)
        report = self.get_status_report()
        for name, info in report.items():
            bar_len = int(info["health_score"] / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(
                "  " + name.ljust(28) +
                bar + " " +
                str(info["health_score"]) + "% " +
                info["status"].upper()
            )
        print("="*55)
        print("  Idle pool size    : " + str(len(self.idle_pool)))
        print("  Total replacements: " + str(self.replacement_count))
        print("="*55 + "\n")