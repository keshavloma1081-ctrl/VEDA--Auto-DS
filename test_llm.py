"""Test VEDA LLM Agents"""
import os
import json
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Build state from previous VEDA run
state = {
    "run_id": run_id,
    "goal": "predict whether a customer will churn. target: Churn",
    "data_profile": {
        "target_column": "Churn",
        "row_count": 7043,
        "col_count": 21,
        "eda_summary": "Telecom dataset with 7043 customers. Churn rate is 26.5%. Key features include tenure, MonthlyCharges, Contract type."
    },
    "model_info": {
        "model_name": "LightGBM",
        "test_metrics": {"auc_roc": 0.8463, "f1_score": 0.5934, "accuracy": 0.8045}
    },
    "explainability": {
        "top_features": ["tenure", "MonthlyCharges", "Contract", "TotalCharges", "InternetService"],
        "explanation_text": "LightGBM model identified tenure and monthly charges as top predictors of churn."
    },
    "cleaning_diff": ["IMPUTED TotalCharges with median", "ENCODED Contract column"],
    "planner_decision_log": []
}

print("="*50)
print("Testing LLM Agents")
print("="*50)

# Test 1 — LLM Chain
print("\n[1/4] LLM Chain Agent...")
from veda.agents.llm.llm_chain import LLMChainAgent
agent1 = LLMChainAgent()
state = agent1.execute(state)

chains = state.get("llm_chains", {})
print("Chains completed: " + str(list(chains.keys())))

# Test 2 — Vector DB
print("\n[2/4] Vector DB Agent...")
from veda.agents.llm.vector_db import VectorDBAgent
agent2 = VectorDBAgent()
state = agent2.execute(state)

vdb = state.get("vector_db", {})
print("Vectors indexed: " + str(vdb.get("num_vectors", 0)))

# Test 3 — RAG Pipeline
print("\n[3/4] RAG Pipeline Agent...")
from veda.agents.llm.rag_pipeline import RAGPipelineAgent
agent3 = RAGPipelineAgent()
state = agent3.execute(state)

rag = state.get("rag_results", [])
print("Questions answered: " + str(len(rag)))

# Test 4 — LLM Evaluation
print("\n[4/4] LLM Evaluation Agent...")
from veda.agents.llm.llm_evaluation import LLMEvaluationAgent
agent4 = LLMEvaluationAgent()
state = agent4.execute(state)

eval_result = state.get("llm_evaluation", {})

print("\n" + "="*50)
print("LLM PIPELINE COMPLETE")
print("="*50)
print("Chains       : " + str(len(chains)))
print("Vectors      : " + str(vdb.get("num_vectors", 0)))
print("RAG answers  : " + str(len(rag)))
print("Hallucinations: " + str(eval_result.get("total_hallucinations", 0)))
print("Quality score : " + str(eval_result.get("avg_quality_score", 0)) + "/5")