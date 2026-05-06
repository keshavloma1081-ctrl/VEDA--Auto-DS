"""Test VEDA Compliance Agents"""
import os
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

d = "outputs"
model_files = sorted([f for f in os.listdir(d) if f.endswith("_model.pkl")])
if not model_files:
    print("No model found — run main.py first")
    exit()

state = {
    "run_id": run_id,
    "goal": "predict whether a customer will default on a loan. target: default",
    "dataset_path": "data/titanic.csv",
    "data_profile": {
        "target_column": "Survived",
        "row_count": 891,
        "col_count": 9
    },
    "model_info": {
        "model_name": "LightGBM",
        "model_path": os.path.join(d, model_files[-1]),
        "mlflow_run_id": "abc123",
        "test_metrics": {"auc_roc": 0.97, "f1_score": 0.85, "accuracy": 0.89}
    },
    "explainability": {
        "top_features": ["Pclass", "Sex_male", "Age", "Fare", "Embarked_S"],
        "explanation_text": "LightGBM model uses passenger class and gender as top predictors.",
        "feature_importance": {"Pclass": 0.35, "Sex_male": 0.28, "Age": 0.18}
    },
    "cleaning_diff": ["IMPUTED Age with median=28", "DROPPED Cabin — 77% nulls"],
    "feature_list": ["Pclass", "Age", "SibSp", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"],
    "pipeline_complete": True,
    "planner_decision_log": []
}

print("="*50)
print("Testing Compliance Agents")
print("="*50)

# Test 1 — PII Detection
print("\n[1/5] PII Detection Agent...")
from veda.agents.special.pii_detection import PIIDetectionAgent
agent1 = PIIDetectionAgent()
state = agent1.execute(state)
pii = state.get("pii_report", {})
print("Risk level  : " + str(pii.get("risk_level")))
print("PII columns : " + str(list(pii.get("pii_columns", {}).keys())))

# Test 2 — Data Masking
print("\n[2/5] Data Masking Agent...")
from veda.agents.special.data_masking import DataMaskingAgent
agent2 = DataMaskingAgent()
state = agent2.execute(state)
masking = state.get("masking_report", {})
print("Changes applied: " + str(len(masking.get("changes_applied", []))))

# Test 3 — GDPR Compliance
print("\n[3/5] GDPR Compliance Agent...")
from veda.agents.special.gdpr_compliance import GDPRComplianceAgent
agent3 = GDPRComplianceAgent()
state = agent3.execute(state)
gdpr = state.get("gdpr_report", {})
print("GDPR status : " + str(gdpr.get("overall_status")))
print("Issues      : " + str(gdpr.get("total_issues")))

# Test 4 — RBI Compliance
print("\n[4/5] RBI Compliance Agent...")
from veda.agents.special.rbi_compliance import RBIComplianceAgent
agent4 = RBIComplianceAgent()
state = agent4.execute(state)
rbi = state.get("rbi_report", {})
print("RBI status : " + str(rbi.get("overall_status")))
print("Issues     : " + str(rbi.get("total_issues")))

# Test 5 — Audit Trail
print("\n[5/5] Audit Trail Agent...")
from veda.agents.special.audit_trail import AuditTrailAgent
agent5 = AuditTrailAgent()
state = agent5.execute(state)
audit = state.get("audit_trail", {})
print("Data hash   : " + str(audit.get("data_hash")))
print("Certificate : " + str(state.get("audit_certificate", ""))[:100])

print("\n" + "="*50)
print("COMPLIANCE PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", []):
    print("  " + log)