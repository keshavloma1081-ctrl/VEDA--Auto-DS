"""Test VEDA MLOps Agents"""
import os
import json
from datetime import datetime

# Load state from previous VEDA run
d = "outputs"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Find latest model
model_files = sorted([f for f in os.listdir(d) if f.endswith("_model.pkl")])
feature_files = sorted([f for f in os.listdir(d) if f.endswith("_features.parquet")])

if not model_files:
    print("No model found — run main.py first")
    exit()

model_path = os.path.join(d, model_files[-1])
print("Using model: " + model_path)

state = {
    "run_id": run_id,
    "goal": "predict whether a passenger survived the Titanic. target: Survived",
    "dataset_path": "data/titanic.csv",
    "data_profile": {
        "target_column": "Survived",
        "row_count": 891,
        "col_count": 9
    },
    "model_info": {
        "model_name": "LightGBM",
        "model_path": model_path,
        "test_metrics": {"auc_roc": 0.97, "f1_score": 0.85, "accuracy": 0.89}
    },
    "feature_list": [],
    "planner_decision_log": [],
    "pipeline_complete": True
}

print("\n" + "="*50)
print("Testing MLOps Agents")
print("="*50)

# Test 1 — Model Serving
print("\n[1/5] Model Serving Agent...")
from veda.agents.mlops.model_serving import ModelServingAgent
agent1 = ModelServingAgent()
state = agent1.execute(state)
serving = state.get("serving_info", {})
print("API generated: " + str(serving.get("api_path", "N/A")))
print("Endpoints    : " + str(serving.get("endpoints", [])))

# Test 2 — Drift Detection
print("\n[2/5] Drift Detection Agent...")
from veda.agents.mlops.drift_detection import DriftDetectionAgent
agent2 = DriftDetectionAgent()
state = agent2.execute(state)
drift = state.get("drift_report", {})
print("Drift score      : " + str(drift.get("drift_score", 0)))
print("Needs retraining : " + str(drift.get("needs_retraining", False)))
print("Drifted features : " + str(len(drift.get("drifted_features", []))))

# Test 3 — Retraining
print("\n[3/5] Retraining Agent...")
from veda.agents.mlops.retraining import RetrainingAgent
agent3 = RetrainingAgent()
state = agent3.execute(state)
retrain = state.get("retraining_results", {})
if retrain:
    print("Retrained : " + str(retrain.get("triggered", False)))
    print("New AUC   : " + str(retrain.get("new_auc", "N/A")))
    print("Improved  : " + str(retrain.get("improved", False)))
else:
    print("Retraining skipped — no drift")

# Test 4 — Pipeline Orchestrator
print("\n[4/5] Pipeline Orchestrator Agent...")
from veda.agents.mlops.pipeline_orchestrator import PipelineOrchestratorAgent
agent4 = PipelineOrchestratorAgent()
state = agent4.execute(state)
orch = state.get("orchestration", {})
print("Total runs : " + str(orch.get("total_runs", 0)))
print("Frequency  : " + str(orch.get("schedule", {}).get("frequency", "N/A")))
print("Next run   : " + str(orch.get("schedule", {}).get("next_run", "N/A"))[:10])

# Test 5 — MLOps Monitor
print("\n[5/5] MLOps Monitor Agent...")
from veda.agents.mlops.mlops_monitor import MLOpsMonitorAgent
agent5 = MLOpsMonitorAgent()
state = agent5.execute(state)
monitor = state.get("monitor_results", {})
perf = monitor.get("performance_metrics", {})
print("P99 latency : " + str(perf.get("latency_p99_ms", "N/A")) + "ms")
print("Error rate  : " + str(perf.get("error_rate", "N/A")))
print("Alerts      : " + str(len(monitor.get("alerts", []))))

print("\n" + "="*50)
print("MLOPS PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", []):
    print("  " + log)