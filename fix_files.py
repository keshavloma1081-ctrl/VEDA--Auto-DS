# fix_files.py — rewrites corrupted agent files

feature_engineering_code = '''import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from veda.core.base_agent import BaseAgent

class FeatureEngineeringAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="FeatureEngineeringAgent", domain="ml", version="1.0.0")

    def _load_cleaned_data(self, state):
        outputs_dir = "outputs"
        files = [f for f in os.listdir(outputs_dir) if f.endswith("_cleaned.parquet")]
        if not files:
            files = [f for f in os.listdir(outputs_dir) if f.endswith("_data.parquet")]
        latest = sorted(files)[-1]
        return pd.read_parquet(os.path.join(outputs_dir, latest))

    def _drop_useless_cols(self, df, target_col=None):
        changes = []
        to_drop = []
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].nunique() == len(df):
                to_drop.append(col)
                changes.append(f"DROPPED {col} — ID column")
            elif df[col].nunique() == 1:
                to_drop.append(col)
                changes.append(f"DROPPED {col} — constant")
        if to_drop:
            df = df.drop(columns=to_drop)
        return df, changes

    def _encode_categoricals(self, df, target_col=None):
        changes = []
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                changes.append(f"ONE-HOT encoded {col}")
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                changes.append(f"LABEL encoded {col}")
        return df, changes

    def _encode_target(self, df, target_col):
        changes = []
        if target_col and target_col in df.columns:
            if df[target_col].dtype == "object":
                le = LabelEncoder()
                df[target_col] = le.fit_transform(df[target_col].astype(str))
                changes.append(f"ENCODED target {target_col}")
        return df, changes

    def _scale_numerics(self, df, target_col=None):
        changes = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            changes.append(f"SCALED {len(numeric_cols)} numeric columns")
        return df, changes

    def run(self, state):
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        self.log("Loading cleaned dataset...")
        df = self._load_cleaned_data(state)
        self.log(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
        all_changes = []
        df, c = self._drop_useless_cols(df, target_col)
        all_changes.extend(c)
        df, c = self._encode_categoricals(df, target_col)
        all_changes.extend(c)
        df, c = self._encode_target(df, target_col)
        all_changes.extend(c)
        df, c = self._scale_numerics(df, target_col)
        all_changes.extend(c)
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        features_path = f"outputs/{run_id}_features.parquet"
        df.to_parquet(features_path, index=False)
        feature_cols = [c for c in df.columns if c != target_col]
        state["feature_list"] = feature_cols
        state.setdefault("planner_decision_log", []).append(
            f"[{datetime.now().isoformat()}] FeatureEngineeringAgent: {len(feature_cols)} features ready"
        )
        self.log("FEATURE ENGINEERING COMPLETE")
        self.log(f"Features: {feature_cols}")
        return state
'''

graph_code = '''from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_veda_graph():
    graph = StateGraph(dict)
    from veda.agents.core_pipeline.planner import PlannerAgent
    from veda.agents.core_pipeline.ingest import IngestAgent
    from veda.agents.core_pipeline.eda import EDAAgent
    from veda.agents.core_pipeline.cleaning import CleaningAgent
    from veda.agents.core_pipeline.feature_engineering import FeatureEngineeringAgent
    planner = PlannerAgent()
    ingest = IngestAgent()
    eda = EDAAgent()
    cleaning = CleaningAgent()
    features = FeatureEngineeringAgent()
    graph.add_node("planner", planner.execute)
    graph.add_node("ingest", ingest.execute)
    graph.add_node("eda", eda.execute)
    graph.add_node("cleaning", cleaning.execute)
    graph.add_node("feature_engineering", features.execute)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "ingest")
    graph.add_edge("ingest", "eda")
    graph.add_edge("eda", "cleaning")
    graph.add_edge("cleaning", "feature_engineering")
    graph.add_edge("feature_engineering", END)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

def run_veda(goal, dataset_path):
    print("\\n==================================================")
    print("  VEDA - Autonomous Data Science System")
    print("==================================================")
    print(f"  Goal    : {goal}")
    print(f"  Dataset : {dataset_path}")
    print("==================================================\\n")
    initial_state = {
        "goal": goal, "dataset_path": dataset_path,
        "current_step": 0, "planner_decision_log": [],
        "cleaning_diff": [], "feature_list": [],
        "agent_health_registry": {}, "pipeline_complete": False,
        "pipeline_failed": False, "human_review_required": False,
        "guard_status": {"output_validation_passed": False,
            "fact_grounding_passed": False, "self_critique_passed": False,
            "metric_consistency_passed": False, "hallucination_count": 0,
            "corrections_made": []},
        "outputs": {"dashboard_path": None, "executive_report_path": None,
            "technical_report_path": None, "model_card_path": None,
            "narrative_text": None, "executive_summary": None}
    }
    veda = build_veda_graph()
    config = {"configurable": {"thread_id": "veda-run-1"}}
    result = veda.invoke(initial_state, config=config)
    print("\\n==================================================")
    print("  VEDA Pipeline Complete")
    print("==================================================")
    return result
'''

with open("veda/agents/core_pipeline/feature_engineering.py", "w") as f:
    f.write(feature_engineering_code)
print("feature_engineering.py written")

with open("veda/core/graph.py", "w") as f:
    f.write(graph_code)
print("graph.py written")

print("All files fixed!")