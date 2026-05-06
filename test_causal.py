"""Test VEDA Causal AI Agents"""
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

state = {
    "run_id": run_id,
    "goal": "predict whether a passenger survived the Titanic. target: Survived",
    "data_profile": {"target_column": "Survived", "row_count": 891},
    "model_info": {
        "model_name": "LightGBM",
        "test_metrics": {"auc_roc": 0.97, "f1_score": 0.85}
    },
    "planner_decision_log": []
}

print("="*50)
print("Testing Causal AI Agents")
print("="*50)

print("\n[1/5] Causal Graph Agent...")
from veda.agents.rag.causal_graph import CausalGraphAgent
agent1 = CausalGraphAgent()
state = agent1.execute(state)
graph = state.get("causal_graph", {})
dag = graph.get("dag", {})
print("Nodes : " + str(len(dag.get("nodes", []))))
print("Edges : " + str(len(dag.get("edges", []))))

print("\n[2/5] Uplift Model Agent...")
from veda.agents.rag.uplift_model import UpliftModelAgent
agent2 = UpliftModelAgent()
state = agent2.execute(state)
uplift = state.get("uplift_results", {})
print("S-CATE : " + str(uplift.get("s_learner", {}).get("avg_cate", "N/A")))
print("Positive uplift: " + str(uplift.get("s_learner", {}).get("positive_uplift_pct", "N/A")) + "%")

print("\n[3/5] A/B Testing Agent...")
from veda.agents.rag.ab_testing import ABTestingAgent
agent3 = ABTestingAgent()
state = agent3.execute(state)
ab = state.get("ab_results", {})
print("Lift           : " + str(ab.get("summary", {}).get("relative_lift_pct", "N/A")) + "%")
print("Recommendation : " + str(ab.get("bayesian", {}).get("recommendation", "N/A")))

print("\n[4/5] Causal Inference Agent...")
from veda.agents.rag.causal_inference import CausalInferenceAgent
agent4 = CausalInferenceAgent()
state = agent4.execute(state)
ci = state.get("causal_inference", {})
print("Naive ATE : " + str(ci.get("naive_ate", {}).get("ate", "N/A")))
print("PSM ATE   : " + str(ci.get("psm_ate", {}).get("ate", "N/A")))

print("\n[5/5] Causal Report Agent...")
from veda.agents.rag.causal_report import CausalReportAgent
agent5 = CausalReportAgent()
state = agent5.execute(state)
summary = state.get("causal_summary", {})
print("Report saved: " + str(summary.get("report_path", "N/A")))

print("\n" + "="*50)
print("CAUSAL AI PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", []):
    print("  " + log)