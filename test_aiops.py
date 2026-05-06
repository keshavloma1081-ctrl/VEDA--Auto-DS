"""Test VEDA AIOps Agents"""
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

state = {
    "run_id": run_id,
    "goal": "monitor VEDA autonomous pipeline",
    "planner_decision_log": []
}

print("="*50)
print("Testing AIOps Agents")
print("="*50)

# Test 1 — Log Analysis
print("\n[1/5] Log Analysis Agent...")
from veda.agents.aiops.log_analysis import LogAnalysisAgent
agent1 = LogAnalysisAgent()
state = agent1.execute(state)
logs = state.get("log_analysis", {})
print("Total logs   : " + str(logs.get("total_logs", 0)))
print("Error rate   : " + str(logs.get("error_rate", {}).get("error_rate", 0)) + "%")
print("Health score : " + str(logs.get("error_rate", {}).get("health_score", 0)))

# Test 2 — Anomaly Detection
print("\n[2/5] Anomaly Detection Agent...")
from veda.agents.aiops.anomaly_detection import AnomalyDetectionAgent
agent2 = AnomalyDetectionAgent()
state = agent2.execute(state)
anomalies = state.get("anomaly_results", {})
print("Total anomalies : " + str(anomalies.get("total_anomalies", 0)))
print("Alerts          : " + str(anomalies.get("alert_count", 0)))
print("Critical        : " + str(anomalies.get("critical_alerts", 0)))

# Test 3 — Root Cause Analysis
print("\n[3/5] Root Cause Analysis Agent...")
from veda.agents.aiops.root_cause import RootCauseAgent
agent3 = RootCauseAgent()
state = agent3.execute(state)
rca = state.get("rca_report", {})
print("Root cause  : " + str(rca.get("root_cause_analysis", {}).get("primary_root_cause", "N/A"))[:80])
print("Confidence  : " + str(rca.get("root_cause_analysis", {}).get("confidence", "N/A")))

# Test 4 — Auto Healing
print("\n[4/5] Auto Healing Agent...")
from veda.agents.aiops.auto_healing import AutoHealingAgent
agent4 = AutoHealingAgent()
state = agent4.execute(state)
healing = state.get("healing_report", {})
print("Actions applied : " + str(healing.get("applied_count", 0)))
print("Actions pending : " + str(healing.get("pending_count", 0)))

# Test 5 — AIOps Monitor
print("\n[5/5] AIOps Monitor Agent...")
from veda.agents.aiops.aiops_monitor import AIOpsMonitorAgent
agent5 = AIOpsMonitorAgent()
state = agent5.execute(state)
monitor = state.get("aiops_monitor", {})
health = monitor.get("health", {})
sla = monitor.get("sla", {})
print("Health score : " + str(health.get("overall_score", 0)) + "/100")
print("Grade        : " + str(health.get("grade", "N/A")))
print("SLA status   : " + str(sla.get("status", "N/A")))

print("\n" + "="*50)
print("AIOPS PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", []):
    print("  " + log)