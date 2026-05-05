"""
VEDA — Autonomous Data Science System
core/graph.py — LangGraph pipeline definition

This is the skeleton of the AutoDS pipeline.
Each agent is a node. Edges define the flow.
State is shared via VEDAState.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from veda.core.state import VEDAState


def build_veda_graph():
    """
    Builds and compiles the full VEDA LangGraph pipeline.
    Agents are added here as nodes as we build them.
    """

    # ── Initialise graph with VEDAState ──────────────────────
    graph = StateGraph(dict)

    # ── Import agents ────────────────────────────────────────
    # We import here to avoid circular imports
    # Add more agents here as we build them sprint by sprint
    from veda.agents.core_pipeline.planner import PlannerAgent
    from veda.agents.core_pipeline.ingest import IngestAgent

    # ── Instantiate agents ───────────────────────────────────
    planner = PlannerAgent()
    ingest = IngestAgent()

    # ── Register nodes ───────────────────────────────────────
    graph.add_node("planner", planner.execute)
    graph.add_node("ingest", ingest.execute)

    # ── Define edges ─────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner", "ingest")
    graph.add_edge("ingest", END)

    # ── Compile with memory ──────────────────────────────────
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    return compiled


def run_veda(goal: str, dataset_path: str):
    """
    Entry point to run the full VEDA pipeline.
    Pass a goal and dataset path — VEDA does the rest.
    """

    print("\n" + "="*50)
    print("  VEDA — Autonomous Data Science System")
    print("="*50)
    print(f"  Goal        : {goal}")
    print(f"  Dataset     : {dataset_path}")
    print("="*50 + "\n")

    # Build initial state
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

    # Build and run graph
    veda = build_veda_graph()
    config = {"configurable": {"thread_id": "veda-run-1"}}

    result = veda.invoke(initial_state, config=config)

    print("\n" + "="*50)
    print("  VEDA Pipeline Complete")
    print("="*50)

    return result