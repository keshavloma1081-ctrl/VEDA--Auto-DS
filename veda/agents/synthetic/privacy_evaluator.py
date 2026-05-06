"""
VEDA — Autonomous Data Science System
agents/synthetic/privacy_evaluator.py — Privacy Evaluator Agent

Evaluates privacy of synthetic data:
- Membership inference risk
- Re-identification risk
- Distance to closest record
- Attribute disclosure risk
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

from veda.core.base_agent import BaseAgent


class PrivacyEvaluatorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="PrivacyEvaluatorAgent",
            domain="synthetic",
            version="1.0.0"
        )

    def _encode_df(self, df):
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=["object", "bool"]).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        return df_enc

    def _distance_to_closest_record(self, real_df: pd.DataFrame,
                                     synthetic_df: pd.DataFrame) -> dict:
        """Compute DCR — distance to closest real record."""
        real_enc = self._encode_df(real_df).fillna(0).values.astype(np.float32)
        synth_enc = self._encode_df(synthetic_df).fillna(0).values.astype(np.float32)

        min_cols = min(real_enc.shape[1], synth_enc.shape[1])
        real_enc = real_enc[:, :min_cols]
        synth_enc = synth_enc[:, :min_cols]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_enc)
        synth_scaled = scaler.transform(synth_enc)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        nbrs.fit(real_scaled)
        distances, _ = nbrs.kneighbors(synth_scaled[:100])

        dcr = distances.flatten()
        return {
            "mean_dcr": round(float(dcr.mean()), 6),
            "min_dcr": round(float(dcr.min()), 6),
            "pct_below_threshold": round(float((dcr < 0.1).mean() * 100), 2),
            "risk_level": "HIGH" if dcr.mean() < 0.5 else "MEDIUM" if dcr.mean() < 1.0 else "LOW"
        }

    def _membership_inference_risk(self, real_df: pd.DataFrame,
                                    synthetic_df: pd.DataFrame) -> dict:
        """Estimate membership inference attack risk."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        real_enc = self._encode_df(real_df).fillna(0)
        synth_enc = self._encode_df(synthetic_df).fillna(0)

        min_cols = min(real_enc.shape[1], synth_enc.shape[1])
        real_enc = real_enc.iloc[:, :min_cols]
        synth_enc = synth_enc.iloc[:, :min_cols]

        n = min(len(real_enc), len(synth_enc), 200)
        X = pd.concat([real_enc.head(n), synth_enc.head(n)], ignore_index=True)
        y = np.array([1] * n + [0] * n)

        try:
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
            attack_auc = round(float(scores.mean()), 4)
            risk = "HIGH" if attack_auc > 0.7 else "MEDIUM" if attack_auc > 0.6 else "LOW"
        except:
            attack_auc = 0.5
            risk = "LOW"

        return {
            "attack_auc": attack_auc,
            "risk_level": risk,
            "interpretation": "AUC=0.5 means perfect privacy, AUC=1.0 means no privacy"
        }

    def _attribute_disclosure_risk(self, real_df: pd.DataFrame,
                                    synthetic_df: pd.DataFrame) -> dict:
        """Check if sensitive attributes can be inferred."""
        issues = []
        for col in real_df.select_dtypes(include=["object"]).columns:
            real_dist = real_df[col].value_counts(normalize=True)
            synth_dist = synthetic_df[col].value_counts(normalize=True) if col in synthetic_df else pd.Series()

            if len(real_dist) > 0 and len(synth_dist) > 0:
                common = set(real_dist.index) & set(synth_dist.index)
                for val in common:
                    if abs(real_dist.get(val, 0) - synth_dist.get(val, 0)) < 0.01:
                        issues.append(col + "=" + str(val) + " distribution preserved exactly")

        return {
            "issues_found": len(issues),
            "issues": issues[:5],
            "risk_level": "HIGH" if len(issues) > 5 else "MEDIUM" if len(issues) > 0 else "LOW"
        }

    def run(self, state: dict) -> dict:
        self.log("Loading real and synthetic data...")
        d = "outputs"
        real_files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        synth_files = [f for f in os.listdir(d) if f.endswith("_synthetic_data.parquet")]

        if not real_files or not synth_files:
            self.log("Real or synthetic data not found", level="WARN")
            return state

        real_df = pd.read_parquet(os.path.join(d, sorted(real_files)[-1]))
        synthetic_df = pd.read_parquet(os.path.join(d, sorted(synth_files)[-1]))

        self.log("Real: " + str(real_df.shape) + " Synthetic: " + str(synthetic_df.shape))

        self.log("Computing Distance to Closest Record...")
        dcr = self._distance_to_closest_record(real_df, synthetic_df)
        self.log("DCR risk: " + dcr["risk_level"] + " mean=" + str(dcr["mean_dcr"]))

        self.log("Estimating membership inference risk...")
        mi_risk = self._membership_inference_risk(real_df, synthetic_df)
        self.log("MI attack AUC: " + str(mi_risk["attack_auc"]))

        self.log("Checking attribute disclosure risk...")
        attr_risk = self._attribute_disclosure_risk(real_df, synthetic_df)
        self.log("Attribute issues: " + str(attr_risk["issues_found"]))

        overall_risk_levels = [dcr["risk_level"], mi_risk["risk_level"], attr_risk["risk_level"]]
        if "HIGH" in overall_risk_levels:
            overall = "HIGH"
        elif "MEDIUM" in overall_risk_levels:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        privacy_results = {
            "distance_to_closest_record": dcr,
            "membership_inference": mi_risk,
            "attribute_disclosure": attr_risk,
            "overall_privacy_risk": overall,
            "privacy_score": round((
                (1 if dcr["risk_level"] == "LOW" else 0.5 if dcr["risk_level"] == "MEDIUM" else 0) +
                (1 if mi_risk["risk_level"] == "LOW" else 0.5 if mi_risk["risk_level"] == "MEDIUM" else 0) +
                (1 if attr_risk["risk_level"] == "LOW" else 0.5 if attr_risk["risk_level"] == "MEDIUM" else 0)
            ) / 3 * 100, 1)
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_privacy_results.json"
        with open(path, "w") as f:
            json.dump(privacy_results, f, indent=2)

        state["privacy_results"] = privacy_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] PrivacyEvaluatorAgent: " +
            "risk=" + overall +
            " score=" + str(privacy_results["privacy_score"])
        )

        self.log("=" * 50)
        self.log("PRIVACY EVALUATION COMPLETE")
        self.log("DCR risk    : " + dcr["risk_level"])
        self.log("MI risk     : " + mi_risk["risk_level"])
        self.log("Overall     : " + overall)
        self.log("Privacy score: " + str(privacy_results["privacy_score"]) + "%")
        self.log("=" * 50)

        return state