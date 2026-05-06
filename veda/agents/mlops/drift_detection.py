"""
VEDA — Autonomous Data Science System
agents/mlops/drift_detection.py — Data Drift Detection Agent
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

from veda.core.base_agent import BaseAgent


class DriftDetectionAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="DriftDetectionAgent",
            domain="mlops",
            version="1.0.0"
        )
        self.PSI_LOW = 0.1
        self.PSI_HIGH = 0.2

    def _load_reference_data(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _simulate_production_data(self, reference_df, drift_factor=0.1):
        prod_df = reference_df.copy()
        numeric_cols = prod_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:
            noise = np.random.normal(
                drift_factor * prod_df[col].std(),
                drift_factor * prod_df[col].std(),
                len(prod_df)
            )
            prod_df[col] = prod_df[col] + noise
        return prod_df

    def _compute_psi(self, reference, production, bins=10):
        ref_clean = reference[~np.isnan(reference)]
        prod_clean = production[~np.isnan(production)]
        if len(ref_clean) == 0 or len(prod_clean) == 0:
            return 0.0
        breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        ref_counts, _ = np.histogram(ref_clean, bins=breakpoints)
        prod_counts, _ = np.histogram(prod_clean, bins=breakpoints)
        ref_pct = ref_counts / len(ref_clean) + 1e-6
        prod_pct = prod_counts / len(prod_clean) + 1e-6
        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        return round(float(psi), 6)

    def _ks_test(self, reference, production):
        ref_clean = reference[~np.isnan(reference)]
        prod_clean = production[~np.isnan(production)]
        if len(ref_clean) < 5 or len(prod_clean) < 5:
            return {"statistic": 0.0, "p_value": 1.0, "drifted": False}
        stat, p_value = stats.ks_2samp(ref_clean, prod_clean)
        return {
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "drifted": bool(p_value < 0.05)
        }

    def _chi_square_test(self, reference, production):
        all_cats = set(reference.unique()) | set(production.unique())
        ref_counts = reference.value_counts()
        prod_counts = production.value_counts()
        ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_cats]) + 1
        prod_freq = np.array([prod_counts.get(cat, 0) for cat in all_cats]) + 1
        ref_expected = ref_freq / ref_freq.sum() * prod_freq.sum()
        try:
            stat, p_value = stats.chisquare(prod_freq, ref_expected)
            return {
                "statistic": round(float(stat), 6),
                "p_value": round(float(p_value), 6),
                "drifted": bool(p_value < 0.05)
            }
        except:
            return {"statistic": 0.0, "p_value": 1.0, "drifted": False}

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading reference data...")
        reference_df = self._load_reference_data(state)

        if reference_df is None:
            self.log("No reference data found", level="WARN")
            return state

        self.log("Reference data: " + str(reference_df.shape))

        self.log("Simulating production data with drift...")
        production_df = self._simulate_production_data(reference_df, drift_factor=0.15)

        feature_cols = [c for c in reference_df.columns if c != target_col]
        numeric_cols = reference_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = reference_df[feature_cols].select_dtypes(include=["object", "bool"]).columns.tolist()

        drift_results = {}
        drifted_features = []

        self.log("Computing PSI and KS tests for " + str(len(numeric_cols)) + " numeric features...")
        for col in numeric_cols[:15]:
            ref_vals = reference_df[col].values
            prod_vals = production_df[col].values
            psi = self._compute_psi(ref_vals, prod_vals)
            ks = self._ks_test(ref_vals, prod_vals)
            psi_status = "stable" if psi < self.PSI_LOW else "warning" if psi < self.PSI_HIGH else "drifted"
            drift_results[col] = {
                "type": "numeric",
                "psi": psi,
                "psi_status": psi_status,
                "ks_test": ks
            }
            if psi > self.PSI_LOW or ks["drifted"]:
                drifted_features.append(col)

        self.log("Running chi-square tests for " + str(len(cat_cols)) + " categorical features...")
        for col in cat_cols[:5]:
            chi2 = self._chi_square_test(
                reference_df[col].fillna("missing"),
                production_df[col].fillna("missing")
            )
            drift_results[col] = {"type": "categorical", "chi2_test": chi2}
            if chi2["drifted"]:
                drifted_features.append(col)

        drift_score = float(len(drifted_features) / len(feature_cols)) if feature_cols else 0.0
        needs_retraining = bool(drift_score > 0.3)

        summary = {
            "reference_shape": list(reference_df.shape),
            "production_shape": list(production_df.shape),
            "features_analyzed": len(drift_results),
            "drifted_features": drifted_features,
            "drift_score": round(drift_score, 4),
            "needs_retraining": needs_retraining,
            "detailed_results": drift_results
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_path = "outputs/" + run_id + "_drift_report.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

        state["drift_report"] = summary
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] DriftDetectionAgent: drift_score=" +
            str(drift_score) + " drifted=" + str(len(drifted_features)) +
            " retrain=" + str(needs_retraining)
        )

        self.log("=" * 50)
        self.log("DRIFT DETECTION COMPLETE")
        self.log("Features analyzed : " + str(len(drift_results)))
        self.log("Drifted features  : " + str(len(drifted_features)))
        self.log("Drift score       : " + str(round(drift_score, 4)))
        self.log("Needs retraining  : " + str(needs_retraining))
        self.log("=" * 50)

        return state