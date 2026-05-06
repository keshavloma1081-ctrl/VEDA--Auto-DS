# Fix 1 — drift_detection.py bool serialization
with open("veda/agents/mlops/drift_detection.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    '"needs_retraining": needs_retraining,',
    '"needs_retraining": bool(needs_retraining),'
)

with open("veda/agents/mlops/drift_detection.py", "w", encoding="utf-8") as f:
    f.write(content)
print("drift_detection.py fixed!")

# Fix 2 — mlops_monitor.py division by zero
with open("veda/agents/mlops/mlops_monitor.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    '"positive_rate": round(sum(p["pred"] for p in predictions) / len(predictions), 4),',
    '"positive_rate": round(sum(p["pred"] for p in predictions) / max(len(predictions), 1), 4),'
)

content = content.replace(
    '"avg_confidence": round(np.mean([p["proba"] for p in predictions]), 4)',
    '"avg_confidence": round(float(np.mean([p["proba"] for p in predictions])) if predictions else 0.0, 4)'
)

with open("veda/agents/mlops/mlops_monitor.py", "w", encoding="utf-8") as f:
    f.write(content)
print("mlops_monitor.py fixed!")