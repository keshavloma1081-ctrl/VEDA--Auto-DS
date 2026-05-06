"""
VEDA — Autonomous Data Science System
idle_pool/idle_agents.py — Idle Agent Reserve Pool

Pre-warmed agents sitting in reserve.
Ready to replace any degraded active agent instantly.
"""

from veda.core.base_agent import BaseAgent
from veda.core.state import VEDAState


class IdlePlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IdlePlanner", domain="orchestration", version="1.0.0")

    def run(self, state: dict) -> dict:
        from veda.agents.core_pipeline.planner import PlannerAgent
        self.log("Idle Planner activated — replacing MasterPlanner")
        real = PlannerAgent()
        return real.run(state)


class IdleIngestAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IdleIngest", domain="data_engineering", version="1.0.0")

    def run(self, state: dict) -> dict:
        from veda.agents.core_pipeline.ingest import IngestAgent
        self.log("Idle Ingest activated — replacing DataIngest")
        real = IngestAgent()
        return real.run(state)


class IdleEDAAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IdleEDA", domain="ml", version="1.0.0")

    def run(self, state: dict) -> dict:
        from veda.agents.core_pipeline.eda import EDAAgent
        self.log("Idle EDA activated — replacing EDAAgent")
        real = EDAAgent()
        return real.run(state)


class IdleCleaningAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IdleCleaning", domain="data_engineering", version="1.0.0")

    def run(self, state: dict) -> dict:
        from veda.agents.core_pipeline.cleaning import CleaningAgent
        self.log("Idle Cleaning activated — replacing CleaningAgent")
        real = CleaningAgent()
        return real.run(state)


class IdleTrainingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IdleTraining", domain="ml", version="1.0.0")

    def run(self, state: dict) -> dict:
        from veda.agents.core_pipeline.training import TrainingAgent
        self.log("Idle Training activated — replacing TrainingAgent")
        real = TrainingAgent()
        return real.run(state)


class IdleEvaluationAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IdleEvaluation", domain="ml", version="1.0.0")

    def run(self, state: dict) -> dict:
        from veda.agents.core_pipeline.evaluation import EvaluationAgent
        self.log("Idle Evaluation activated — replacing EvaluationAgent")
        real = EvaluationAgent()
        return real.run(state)


class IdleGuardAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IdleGuard", domain="guard", version="1.0.0")

    def run(self, state: dict) -> dict:
        self.log("Idle Guard activated — standing by")
        return state


class IdleReportAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IdleReport", domain="reports", version="1.0.0")

    def run(self, state: dict) -> dict:
        from veda.agents.reports.report import ReportAgent
        self.log("Idle Report activated — replacing ReportAgent")
        real = ReportAgent()
        return real.run(state)


def build_idle_pool() -> dict:
    """
    Build and return the full idle agent pool.
    Keys are the domain names they can cover.
    """
    return {
        "orchestration": IdlePlannerAgent(),
        "data_engineering": IdleIngestAgent(),
        "ml": IdleEDAAgent(),
        "cleaning": IdleCleaningAgent(),
        "training": IdleTrainingAgent(),
        "evaluation": IdleEvaluationAgent(),
        "guard": IdleGuardAgent(),
        "reports": IdleReportAgent(),
    }