graph_code = """from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_veda_graph():
    graph = StateGraph(dict)
    from veda.agents.core_pipeline.planner import PlannerAgent
    from veda.agents.core_pipeline.ingest import IngestAgent
    from veda.agents.core_pipeline.eda import EDAAgent
    from veda.agents.core_pipeline.cleaning import CleaningAgent
    from veda.agents.core_pipeline.feature_engineering import FeatureEngineeringAgent
    from veda.agents.core_pipeline.model_selection import ModelSelectionAgent
    from veda.agents.core_pipeline.training import TrainingAgent
    planner = PlannerAgent()
    ingest = IngestAgent()
    eda = EDAAgent()
    cleaning = CleaningAgent()
    features = FeatureEngineeringAgent()
    model_sel = ModelSelectionAgent()
    training = TrainingAgent()
    graph.add_node("planner", planner.execute)
    graph.add_node("ingest", ingest.execute)
    graph.add_node("eda", eda.execute)
    graph.add_node("cleaning", cleaning.execute)
    graph.add_node("feature_engineering", features.execute)
    graph.add_node("model_selection", model_sel.execute)
    graph.add_node("training", training.execute)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "ingest")
    graph.add_edge("ingest", "eda")
    graph.add_edge("eda", "cleaning")
    graph.add_edge("cleaning", "feature_engineering")
    graph.add_edge("feature_engineering", "model_selection")
    graph.add_edge("model_selection", "training")
    graph.add_edge("training", END)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

def run_veda(goal, dataset_path):
    print("\\n==================================================")
    print("  VEDA - Autonomous Data Science System")
    print("==================================================")
    print("  Goal    : " + goal)
    print("  Dataset : " + dataset_path)
    print("==================================================\\n")
    initial_state = {
        "goal": goal, "dataset_path": dataset_path,
        "current_step": 0, "planner_decision_log": [],
        "cleaning_diff": [], "feature_list": [],
        "agent_health_registry": {}, "pipeline_complete": False,
        "pipeline_failed": False, "human_review_required": False,
        "model_info": {},
        "guard_status": {"output_validation_passed": False,
            "fact_grounding_passed": False, "self_critique_passed": False,
            "metric_consistency_passed": False, "hallucination_count": 0,
            "corrections_made": []},
        "outputs": {"dashboard_path": None, "executive_report_path": None,
            "technical_report_path": None, "model_card_path": None,
            "narrative_text": None, "executive_summary": None}
    }
    veda = build_veda_graph()
    config = {"configurable": {"thread_id": "veda-run-1"}}
    result = veda.invoke(initial_state, config=config)
    print("\\n==================================================")
    print("  VEDA Pipeline Complete")
    print("==================================================")
    return result
"""

with open("veda/core/graph.py", "w", encoding="utf-8") as f:
    f.write(graph_code)
print("graph.py updated with TrainingAgent!")