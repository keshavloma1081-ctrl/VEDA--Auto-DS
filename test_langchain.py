"""Test VEDA LangChain Agents"""
import os
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
d = "outputs"
model_files = sorted([f for f in os.listdir(d) if f.endswith("_model.pkl")])

state = {
    "run_id": run_id,
    "goal": "predict whether a passenger survived the Titanic. target: Survived",
    "dataset_path": "data/titanic.csv",
    "data_profile": {"target_column": "Survived", "row_count": 891, "col_count": 9},
    "model_info": {
        "model_name": "LightGBM",
        "model_path": os.path.join(d, model_files[-1]) if model_files else "",
        "test_metrics": {"auc_roc": 0.97, "f1_score": 0.85, "accuracy": 0.89}
    },
    "feature_list": ["Pclass", "Age", "SibSp", "Fare", "Sex_male"],
    "drift_report": {"drift_score": 0.06, "needs_retraining": False},
    "gdpr_report": {"overall_status": "NEEDS_ATTENTION"},
    "planner_decision_log": [
        "MasterPlanner: classification task",
        "DataIngest: 891 rows loaded",
        "EDAAgent: analysis complete",
        "TrainingAgent: LightGBM AUC=0.97"
    ]
}

print("="*50)
print("Testing LangChain Agents")
print("="*50)

# Test 1 — Chain Builder
print("\n[1/7] Chain Builder Agent...")
from veda.agents.langchain.chain_builder import ChainBuilderAgent
agent1 = ChainBuilderAgent()
state = agent1.execute(state)
chains = state.get("lc_chains", {})
print("Chains built : " + str(list(chains.keys())))
print("Task type    : " + str(chains.get("classification", {}).get("category", "N/A")))

# Test 2 — Memory Agent
print("\n[2/7] Memory Agent...")
from veda.agents.langchain.memory_agent import MemoryAgent
agent2 = MemoryAgent()
state = agent2.execute(state)
memory = state.get("lc_memory", {})
print("Short-term : " + str(len(memory.get("short_term", []))))
print("Long-term  : " + str(len(memory.get("long_term", {}))))

# Test 3 — Tool Agent
print("\n[3/7] Tool Agent...")
from veda.agents.langchain.tool_agent import ToolAgent
agent3 = ToolAgent()
state = agent3.execute(state)
tools = state.get("lc_tool_results", [])
print("Tool calls : " + str(len(tools)))

# Test 4 — Workflow Agent
print("\n[4/7] Workflow Agent...")
from veda.agents.langchain.workflow_agent import WorkflowAgent
agent4 = WorkflowAgent()
state = agent4.execute(state)
workflow = state.get("lc_workflow", {})
print("Steps completed : " + str(len(workflow.get("sequential_results", []))))
print("Branch decisions: " + str(workflow.get("branch_decisions", {})))

# Test 5 — Data Loader
print("\n[5/7] Data Loader Agent...")
from veda.agents.langchain.data_loader import DataLoaderAgent
agent5 = DataLoaderAgent()
state = agent5.execute(state)
docs = state.get("lc_documents", {})
print("Documents loaded: " + str(docs.get("total_documents", 0)))

# Test 6 — Retriever Agent
print("\n[6/7] Retriever Agent...")
from veda.agents.langchain.retriever_agent import RetrieverAgent
agent6 = RetrieverAgent()
state = agent6.execute(state)
retrieval = state.get("lc_retrieval", {})
print("Queries answered: " + str(len(retrieval)))

# Test 7 — LangSmith Agent
print("\n[7/7] LangSmith Agent...")
from veda.agents.langchain.langsmith_agent import LangSmithAgent
agent7 = LangSmithAgent()
state = agent7.execute(state)
traces = state.get("lc_traces", {})
stats = traces.get("stats", {})
print("Total calls : " + str(stats.get("total_calls", 0)))
print("Avg latency : " + str(stats.get("avg_latency_ms", 0)) + "ms")

# Test 8 — LC Evaluator
print("\n[8/7] LC Evaluator Agent...")
from veda.agents.langchain.lc_evaluator import LCEvaluatorAgent
agent8 = LCEvaluatorAgent()
state = agent8.execute(state)
eval_result = state.get("lc_evaluation", {})
print("Avg score : " + str(eval_result.get("avg_quality_score", 0)) + "/5")
print("Grade     : " + str(eval_result.get("overall_grade", "N/A")))

print("\n" + "="*50)
print("LANGCHAIN PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", [])[-8:]:
    print("  " + log)