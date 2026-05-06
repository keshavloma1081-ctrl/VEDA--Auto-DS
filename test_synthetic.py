"""Test VEDA Synthetic Data Agents"""
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

state = {
    "run_id": run_id,
    "goal": "generate synthetic training data for credit scoring",
    "data_profile": {"target_column": "Survived", "row_count": 891},
    "planner_decision_log": []
}

print("="*50)
print("Testing Synthetic Data Agents")
print("="*50)

print("\n[1/5] Synthetic Tabular Agent...")
from veda.agents.synthetic.synthetic_tabular import SyntheticTabularAgent
agent1 = SyntheticTabularAgent()
state = agent1.execute(state)
synthetic = state.get("synthetic_results", {})
print("Method     : " + str(synthetic.get("method")))
print("Real rows  : " + str(synthetic.get("real_rows")))
print("Synth rows : " + str(synthetic.get("synthetic_rows")))

print("\n[2/5] Data Augmentation Agent...")
from veda.agents.synthetic.data_augmentation import DataAugmentationAgent
agent2 = DataAugmentationAgent()
state = agent2.execute(state)
aug = state.get("augmentation_results", {})
print("Original  : " + str(aug.get("original_shape")))
print("After SMOTE: " + str(aug.get("final_train_size")))

print("\n[3/5] Privacy Evaluator Agent...")
from veda.agents.synthetic.privacy_evaluator import PrivacyEvaluatorAgent
agent3 = PrivacyEvaluatorAgent()
state = agent3.execute(state)
privacy = state.get("privacy_results", {})
print("Overall risk  : " + str(privacy.get("overall_privacy_risk")))
print("Privacy score : " + str(privacy.get("privacy_score")) + "%")

print("\n[4/5] Statistical Fidelity Agent...")
from veda.agents.synthetic.statistical_fidelity import StatisticalFidelityAgent
agent4 = StatisticalFidelityAgent()
state = agent4.execute(state)
fidelity = state.get("fidelity_results", {})
print("Fidelity score : " + str(fidelity.get("fidelity_score")) + "%")
print("Grade          : " + str(fidelity.get("grade")))

print("\n[5/5] Synthetic Report Agent...")
from veda.agents.synthetic.synthetic_report import SyntheticReportAgent
agent5 = SyntheticReportAgent()
state = agent5.execute(state)
report = state.get("synthetic_report", {})
print("Report saved: " + str(report.get("report_path")))

print("\n" + "="*50)
print("SYNTHETIC DATA PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", []):
    print("  " + log)