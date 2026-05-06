import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from veda.core.base_agent import BaseAgent

class FeatureEngineeringAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="FeatureEngineeringAgent", domain="ml", version="1.0.0")

    def _load_cleaned_data(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _drop_useless_cols(self, df, target_col=None):
        changes = []
        to_drop = []
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].nunique() == len(df):
                to_drop.append(col)
                changes.append("DROPPED " + col + " ID column")
            elif df[col].nunique() == 1:
                to_drop.append(col)
                changes.append("DROPPED " + col + " constant")
        if to_drop:
            df = df.drop(columns=to_drop)
        return df, changes

    def _encode_categoricals(self, df, target_col=None):
        changes = []
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        for col in cat_cols:
            if df[col].nunique() <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                changes.append("ONE-HOT encoded " + col)
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                changes.append("LABEL encoded " + col)
        return df, changes

    def _encode_target(self, df, target_col):
        changes = []
        if target_col and target_col in df.columns:
            if df[target_col].dtype == "object":
                le = LabelEncoder()
                df[target_col] = le.fit_transform(df[target_col].astype(str))
                changes.append("ENCODED target " + str(target_col))
        return df, changes

    def _scale_numerics(self, df, target_col=None):
        changes = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            changes.append("SCALED " + str(len(numeric_cols)) + " numeric columns")
        return df, changes

    def run(self, state):
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        self.log("Loading cleaned dataset...")
        df = self._load_cleaned_data(state)
        self.log("Loaded " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " columns")
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
        features_path = "outputs/" + run_id + "_features.parquet"
        df.to_parquet(features_path, index=False)
        feature_cols = [col for col in df.columns if col != target_col]
        state["feature_list"] = feature_cols
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] FeatureEngineeringAgent: " + str(len(feature_cols)) + " features ready"
        )
        self.log("FEATURE ENGINEERING COMPLETE")
        self.log("Features: " + str(feature_cols))
        return state
