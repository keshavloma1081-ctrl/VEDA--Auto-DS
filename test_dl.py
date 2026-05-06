"""Test MLP Agent on Titanic dataset"""
import os
import pandas as pd
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load features from previous VEDA run
d = "outputs"
files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
if not files:
    print("No features found — run main.py first")
    exit()

df = pd.read_parquet(os.path.join(d, sorted(files)[-1]))
print("Loaded features: " + str(df.shape))

# Save as current run
df.to_parquet("outputs/" + run_id + "_features.parquet", index=False)

state = {
    "run_id": run_id,
    "goal": "predict survival. target: Survived",
    "data_profile": {"target_column": df.columns[-1]},
    "planner_decision_log": []
}

print("\nTesting MLP Agent...")
from veda.agents.deep_learning.mlp import MLPAgent
agent = MLPAgent()
state = agent.execute(state)

print("\nResults:")
results = state.get("dl_results", {})
print("AUC      : " + str(results.get("auc")))
print("F1       : " + str(results.get("f1")))
print("Accuracy : " + str(results.get("accuracy")))
print("Params   : " + str(results.get("total_params")))

print("\nTesting CNN Agent...")
from veda.agents.deep_learning.cnn import CNNAgent
agent2 = CNNAgent()
state = agent2.execute(state)

cnn = state.get("dl_results", {}).get("cnn", {})
print("CNN Results:")
print("AUC      : " + str(cnn.get("auc")))
print("F1       : " + str(cnn.get("f1")))
print("Params   : " + str(cnn.get("total_params")))

print("\nTesting LSTM Agent...")
from veda.agents.deep_learning.lstm import LSTMAgent
agent3 = LSTMAgent()
state = agent3.execute(state)

lstm = state.get("dl_results", {}).get("lstm", {})
print("LSTM Results:")
print("AUC      : " + str(lstm.get("auc")))
print("F1       : " + str(lstm.get("f1")))
print("Params   : " + str(lstm.get("total_params")))

print("\nTesting DL Trainer Agent...")
from veda.agents.deep_learning.trainer import DLTrainerAgent
agent4 = DLTrainerAgent()
state = agent4.execute(state)

print("\nTesting DL Evaluation Agent...")
from veda.agents.deep_learning.dl_evaluation import DLEvaluationAgent
agent5 = DLEvaluationAgent()
state = agent5.execute(state)

print("\n" + "="*50)
print("DEEP LEARNING PIPELINE COMPLETE")
print("="*50)
dl_eval = state.get("dl_evaluation", {})
print("Best DL model : " + str(dl_eval.get("best_model")))
print("Best AUC      : " + str(dl_eval.get("best_auc")))
print("Interpretation: " + str(dl_eval.get("interpretation", ""))[:200])