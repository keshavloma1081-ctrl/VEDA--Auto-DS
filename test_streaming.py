"""Test VEDA Streaming Agents"""
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

state = {
    "run_id": run_id,
    "goal": "monitor real-time transaction stream for fraud detection",
    "planner_decision_log": []
}

print("="*50)
print("Testing Streaming Agents")
print("="*50)

print("\n[1/5] Stream Ingest Agent...")
from veda.agents.streaming.stream_ingest import StreamIngestAgent
agent1 = StreamIngestAgent()
state = agent1.execute(state)
ingest = state.get("stream_ingest", {})
print("Events ingested : " + str(ingest.get("successfully_ingested")))
print("Throughput      : " + str(ingest.get("throughput_per_sec")) + " eps")
print("Failure rate    : " + str(ingest.get("failure_rate")) + "%")

print("\n[2/5] Stream Processor Agent...")
from veda.agents.streaming.stream_processor import StreamProcessorAgent
agent2 = StreamProcessorAgent()
state = agent2.execute(state)
processor = state.get("stream_processor", {})
print("Tumbling windows: " + str(processor.get("tumbling_windows")))
print("Sliding windows : " + str(processor.get("sliding_windows")))
print("Features        : " + str(processor.get("enriched_features")))

print("\n[3/5] Online Learning Agent...")
from veda.agents.streaming.online_learning import OnlineLearningAgent
agent3 = OnlineLearningAgent()
state = agent3.execute(state)
online = state.get("online_learning", {})
print("Best model  : " + str(online.get("best_model")))
print("HT accuracy : " + str(online.get("hoeffding_tree", {}).get("final_accuracy")))
print("Drift       : " + str(online.get("concept_drift", {}).get("drift_detected")))

print("\n[4/5] Stream Anomaly Agent...")
from veda.agents.streaming.stream_anomaly import StreamAnomalyAgent
agent4 = StreamAnomalyAgent()
state = agent4.execute(state)
anomaly = state.get("stream_anomaly", {})
print("EWMA anomalies : " + str(anomaly.get("ewma", {}).get("anomaly_count")))
print("ISO anomalies  : " + str(anomaly.get("isolation_forest", {}).get("anomaly_count")))
print("Alerts         : " + str(len(anomaly.get("alerts", []))))

print("\n[5/5] Stream Monitor Agent...")
from veda.agents.streaming.stream_monitor import StreamMonitorAgent
agent5 = StreamMonitorAgent()
state = agent5.execute(state)
monitor = state.get("stream_monitor", {})
print("Health score : " + str(monitor.get("health", {}).get("overall_score")) + "/100")
print("Grade        : " + str(monitor.get("health", {}).get("grade")))
print("SLA          : " + str(monitor.get("sla", {}).get("status")))

print("\n" + "="*50)
print("STREAMING PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", []):
    print("  " + log)