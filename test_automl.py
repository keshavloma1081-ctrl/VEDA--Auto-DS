"""Test VEDA AutoML Agents"""
from datetime import datetime
import os

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
d = "outputs"
model_files = sorted([f for f in os.listdir(d) if f.endswith("_model.pkl")])

state = {
    "run_id": run_id,
    "goal": "predict whether a passenger survived the Titanic. target: Survived",
    "data_profile": {"target_column": "Survived", "row_count": 891},
    "model_info": {
        "model_name": "LightGBM",
        "model_path": os.path.join(d, model_files[-1]) if model_files else "",
        "test_metrics": {"auc_roc": 0.97, "f1_score": 0.85}
    },
    "planner_decision_log": []
}

print("="*50)
print("Testing AutoML Agents")
print("="*50)

print("\n[1/5] AutoML Search Agent...")
from veda.agents.automl.automl_search import AutoMLSearchAgent
agent1 = AutoMLSearchAgent()
state = agent1.execute(state)
automl = state.get("automl_results", {})
print("Best model : " + str(automl.get("best_estimator")))
print("AUC        : " + str(automl.get("auc")))

print("\n[2/5] Hyperopt Agent...")
from veda.agents.automl.hyperopt_agent import HyperoptAgent
agent2 = HyperoptAgent()
state = agent2.execute(state)
hyperopt = state.get("hyperopt_results", {})
print("Best AUC   : " + str(hyperopt.get("best_auc")))
print("Method     : " + str(hyperopt.get("method")))

print("\n[3/5] Feature Selector Agent...")
from veda.agents.automl.feature_selector import FeatureSelectorAgent
agent3 = FeatureSelectorAgent()
state = agent3.execute(state)
features = state.get("feature_selection", {})
print("Original  : " + str(features.get("original_features")))
print("Selected  : " + str(features.get("final_selected")))
print("Reduction : " + str(features.get("reduction_pct")) + "%")

print("\n[4/5] Model Compression Agent...")
from veda.agents.automl.model_compression import ModelCompressionAgent
agent4 = ModelCompressionAgent()
state = agent4.execute(state)
compression = state.get("compression_results", {})
print("Original size : " + str(compression.get("original_model_size", {}).get("size_kb")) + " KB")
print("After quant   : " + str(compression.get("quantization", {}).get("quantized_size_kb")) + " KB")

print("\n[5/5] AutoML Report Agent...")
from veda.agents.automl.automl_report import AutoMLReportAgent
agent5 = AutoMLReportAgent()
state = agent5.execute(state)
report = state.get("automl_report", {})
print("Readiness : " + str(report.get("deployment_readiness", {}).get("score")) + "%")
print("Summary   : " + str(report.get("summary", ""))[:100])

print("\n" + "="*50)
print("AUTOML PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", []):
    print("  " + log)