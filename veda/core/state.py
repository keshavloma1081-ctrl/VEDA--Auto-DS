"""
VEDA — Autonomous Data Science System
core/state.py — Shared state schema for all 128 agents
Every agent reads from and writes to this single state object.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ── Execution Plan ──────────────────────────────────────────
class ExecutionStep(BaseModel):
    step_number: int
    agent_name: str
    description: str
    status: str = "pending"  # pending | running | done | failed | skipped
    output_key: Optional[str] = None


class ExecutionPlan(BaseModel):
    goal: str
    task_type: str  # classification | regression | timeseries | clustering | nlp
    steps: list[ExecutionStep]
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Data Layer ───────────────────────────────────────────────
class DataProfile(BaseModel):
    row_count: int = 0
    col_count: int = 0
    target_column: Optional[str] = None
    feature_columns: list[str] = []
    null_counts: dict[str, int] = {}
    dtypes: dict[str, str] = {}
    class_balance: Optional[dict] = None
    has_imbalance: bool = False
    has_leakage_risk: bool = False
    eda_summary: Optional[str] = None


# ── Model Layer ──────────────────────────────────────────────
class ModelMetrics(BaseModel):
    auc_roc: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    accuracy: Optional[float] = None


class ModelInfo(BaseModel):
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    pipeline_path: Optional[str] = None
    hyperparameters: dict = {}
    cv_metrics: Optional[ModelMetrics] = None
    test_metrics: Optional[ModelMetrics] = None
    mlflow_run_id: Optional[str] = None
    passed_evaluation: bool = False


# ── Explainability ───────────────────────────────────────────
class ExplainabilityOutput(BaseModel):
    top_features: list[str] = []
    shap_summary_path: Optional[str] = None
    shap_waterfall_paths: list[str] = []
    explanation_text: Optional[str] = None
    leakage_flags: list[str] = []


# ── Guard Layer ──────────────────────────────────────────────
class GuardStatus(BaseModel):
    output_validation_passed: bool = False
    fact_grounding_passed: bool = False
    self_critique_passed: bool = False
    metric_consistency_passed: bool = False
    hallucination_count: int = 0
    corrections_made: list[str] = []


# ── Agent Health ─────────────────────────────────────────────
class AgentHealth(BaseModel):
    agent_name: str
    health_score: float = 100.0   # 0-100
    status: str = "healthy"       # healthy | degraded | critical
    error_count: int = 0
    last_ping: Optional[str] = None
    replacing: bool = False


# ── Outputs ──────────────────────────────────────────────────
class VEDAOutputs(BaseModel):
    dashboard_path: Optional[str] = None
    executive_report_path: Optional[str] = None
    technical_report_path: Optional[str] = None
    model_card_path: Optional[str] = None
    narrative_text: Optional[str] = None
    executive_summary: Optional[str] = None


# ── MASTER STATE ─────────────────────────────────────────────
class VEDAState(BaseModel):
    """
    The single source of truth for the entire VEDA pipeline.
    All 128 agents read from and write to this object.
    LangGraph persists this between agent calls via MemorySaver.
    """

    # Run identity
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    goal: Optional[str] = None
    dataset_path: Optional[str] = None

    # Planning
    execution_plan: Optional[ExecutionPlan] = None
    current_step: int = 0
    planner_decision_log: list[str] = []

    # Data
    data_profile: Optional[DataProfile] = None
    cleaning_diff: list[str] = []
    feature_list: list[str] = []

    # Model
    model_info: Optional[ModelInfo] = None
    benchmark_table: Optional[str] = None

    # Explainability
    explainability: Optional[ExplainabilityOutput] = None

    # Guard
    guard_status: GuardStatus = Field(default_factory=GuardStatus)

    # Agent health (self-healing layer)
    agent_health_registry: dict[str, AgentHealth] = {}

    # Outputs
    outputs: VEDAOutputs = Field(default_factory=VEDAOutputs)

    # Pipeline control
    pipeline_complete: bool = False
    pipeline_failed: bool = False
    failure_reason: Optional[str] = None
    human_review_required: bool = False

    class Config:
        arbitrary_types_allowed = True