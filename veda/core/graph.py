"""
VEDA — Autonomous Data Science System
core/graph.py — LangGraph pipeline definition
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


def build_veda_graph():

    graph = StateGraph(dict)

    # ── Import agents ────────────────────────────────────────
    from veda.agents.core_pipeline.planner import PlannerAgent
    from veda.agents.core_pipeline.ingest import IngestAgent
    from veda.agents.core_pipeline.eda import EDAAgent

    # ── Instantiate agents ───────────────────────────────────
    planner = PlannerAgent()
    ingest = IngestAgent()
    eda = EDAAgent()

    # ── Register nodes ───────────────────────────────────────
    graph.add_node("planner", planner.execute)
    graph.add_node("ingest", ingest.execute)
    graph.add_node("eda", eda.execute)

    # ── Define edges ─────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner", "ingest")
    graph.add_edge("ingest", "eda")
    graph.add_edge("eda", END)

    # ── Compile with memory ──────────────────────────────────
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    return compiled


def run_veda(goal: str, dataset_path: str):

    print("\n" + "="*50)
    print("  VEDA — Autonomous Data Science System")
    print("="*50)
    print(f"  Goal        : {goal}")
    print(f"  Dataset     : {dataset_path}")
    print("="*50 + "\n")

    initial_state = {
        "goal": goal,
        "dataset_path": dataset_path,
        "current_step": 0,
        "planner_decision_log": [],
        "cleaning_diff": [],
        "feature_list": [],
        "agent_health_registry": {},
        "pipeline_complete": False,
        "pipeline_failed": False,
        "human_review_required": False,
        "guard_status": {
            "output_validation_passed": False,
            "fact_grounding_passed": False,
            "self_critique_passed": False,
            "metric_consistency_passed": False,
            "hallucination_count": 0,
            "corrections_made": []
        },
        "outputs": {
            "dashboard_path": None,
            "executive_report_path": None,
            "technical_report_path": None,
            "model_card_path": None,
            "narrative_text": None,
            "executive_summary": None
        }
    }

    veda = build_veda_graph()
    config = {"configurable": {"thread_id": "veda-run-1"}}
    result = veda.invoke(initial_state, config=config)

    print("\n" + "="*50)
    print("  VEDA Pipeline Complete")
    print("="*50)

    return result