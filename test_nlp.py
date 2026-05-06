"""
Test VEDA NLP Pipeline on the churn dataset
"""

import os
import pandas as pd
from datetime import datetime

# Set up initial state
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load churn dataset and add a text column for testing
print("Loading churn dataset...")
df = pd.read_csv("data/churn.csv", nrows=1000)

# Add a synthetic text review column for NLP testing
import random
random.seed(42)

positive = ["great service love it", "excellent product recommend", "very happy satisfied customer"]
negative = ["terrible service awful experience", "disappointed bad quality", "worst product ever avoid"]
neutral = ["okay product average service", "decent nothing special", "standard product works fine"]

df["customer_feedback"] = [
    random.choice(positive + negative + neutral) for _ in range(len(df))
]

# Save as parquet for VEDA
os.makedirs("outputs", exist_ok=True)
data_path = "outputs/" + run_id + "_data.parquet"
df.to_parquet(data_path, index=False)
print("Data saved to: " + data_path)
print("Shape: " + str(df.shape))

# Build state
state = {
    "run_id": run_id,
    "goal": "analyze customer feedback and predict churn. target: Churn",
    "dataset_path": "data/churn.csv",
    "data_profile": {
        "target_column": "Churn",
        "feature_columns": list(df.columns),
        "row_count": len(df),
        "col_count": len(df.columns)
    },
    "planner_decision_log": [],
    "feature_list": []
}

print("\n" + "="*50)
print("Testing NLP Pipeline")
print("="*50)

# Test 1 — Text Preprocessing
print("\n[1/4] Text Preprocessing Agent...")
from veda.agents.nlp.text_preprocessing import TextPreprocessingAgent
agent1 = TextPreprocessingAgent()
state = agent1.execute(state)

# Test 2 — Text Classification
print("\n[2/4] Text Classification Agent...")
from veda.agents.nlp.text_classification import TextClassificationAgent
agent2 = TextClassificationAgent()
state = agent2.execute(state)

# Test 3 — NER Agent
print("\n[3/4] NER Agent...")
from veda.agents.nlp.ner import NERAgent
agent3 = NERAgent()
state = agent3.execute(state)

# Test 4 — Sentiment Analysis
print("\n[4/4] Sentiment Analysis Agent...")
from veda.agents.nlp.sentiment import SentimentAnalysisAgent
agent4 = SentimentAnalysisAgent()
state = agent4.execute(state)

# Test 5 — Summarization
print("\n[5/5] Text Summarization Agent...")
from veda.agents.nlp.summarization import TextSummarizationAgent
agent5 = TextSummarizationAgent()
state = agent5.execute(state)

print("\n" + "="*50)
print("NLP PIPELINE COMPLETE")
print("="*50)
print("Decision log entries: " + str(len(state.get("planner_decision_log", []))))
for log in state.get("planner_decision_log", []):
    print("  " + log)