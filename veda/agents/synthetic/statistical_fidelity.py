"""
VEDA — Autonomous Data Science System
agents/synthetic/statistical_fidelity.py — Statistical Fidelity Agent

Compares real vs synthetic data statistically:
- Column statistics comparison
- KS test per feature
- Correlation matrix comparison
- Distribution similarity score
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent


class StatisticalFidelityAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="StatisticalFidelityAgent",
            domain="synthetic",
            version="1.0.0"
        )

    def _encode_df(self, df):
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=["object", "bool"]).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        return df_enc

    def _compare_column_stats(self, real_df: pd.DataFrame,
                               synthetic_df: pd.DataFrame) -> dict:
        """Compare basic statistics per column."""
        results = {}
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in synthetic_df.columns:
                real_vals = real_df[col].dropna()
                synth_vals = synthetic_df[col].dropna()
                mean_diff = abs(float(real_vals.mean()) - float(synth_vals.mean()))
                std_diff = abs(float(real_vals.std()) - float(synth_vals.std()))
                results[col] = {
                    "real_mean": round(float(real_vals.mean()), 4),
                    "synth_mean": round(float(synth_vals.mean()), 4),
                    "mean_diff": round(mean_diff, 4),
                    "real_std": round(float(real_vals.std()), 4),
                    "synth_std": round(float(synth_vals.std()), 4),
                    "std_diff": round(std_diff, 4),
                    "similar": bool(mean_diff < real_vals.std() * 0.1)
                }
        return results

    def _ks_test_per_column(self, real_df: pd.DataFrame,
                              synthetic_df: pd.DataFrame) -> dict:
        """KS test for each numeric column."""
        results = {}
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in synthetic_df.columns:
                real_vals = real_df[col].dropna().values
                synth_vals = synthetic_df[col].dropna().values
                if len(real_vals) > 0 and len(synth_vals) > 0:
                    stat, p_value = stats.ks_2samp(real_vals, synth_vals)
                    results[col] = {
                        "statistic": round(float(stat), 6),
                        "p_value": round(float(p_value), 6),
                        "distributions_similar": bool(p_value > 0.05)
                    }
        return results

    def _correlation_similarity(self, real_df: pd.DataFrame,
                                 synthetic_df: pd.DataFrame) -> dict:
        """Compare correlation matrices."""
        real_enc = self._encode_df(real_df).fillna(0)
        synth_enc = self._encode_df(synthetic_df).fillna(0)

        common_cols = list(set(real_enc.columns) & set(synth_enc.columns))[:10]

        real_corr = real_enc[common_cols].corr().values
        synth_corr = synth_enc[common_cols].corr().values

        diff = np.abs(real_corr - synth_corr)
        mean_diff = round(float(np.nanmean(diff)), 6)

        return {
            "mean_correlation_diff": mean_diff,
            "max_correlation_diff": round(float(np.nanmax(diff)), 6),
            "correlation_similar": bool(mean_diff < 0.1)
        }

    def _compute_fidelity_score(self, col_stats: dict,
                                 ks_results: dict,
                                 corr_sim: dict) -> float:
        """Compute overall fidelity score."""
        scores = []

        if col_stats:
            similar_cols = sum(1 for v in col_stats.values() if v.get("similar", False))
            scores.append(similar_cols / max(len(col_stats), 1))

        if ks_results:
            similar_dists = sum(1 for v in ks_results.values()
                               if v.get("distributions_similar", False))
            scores.append(similar_dists / max(len(ks_results), 1))

        if corr_sim.get("correlation_similar", False):
            scores.append(1.0)
        else:
            scores.append(max(0, 1 - corr_sim.get("mean_correlation_diff", 1)))

        return round(float(np.mean(scores) * 100), 2) if scores else 0.0

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

        self.log("Comparing column statistics...")
        col_stats = self._compare_column_stats(real_df, synthetic_df)
        similar_cols = sum(1 for v in col_stats.values() if v.get("similar"))
        self.log("Similar columns: " + str(similar_cols) + "/" + str(len(col_stats)))

        self.log("Running KS tests...")
        ks_results = self._ks_test_per_column(real_df, synthetic_df)
        similar_dists = sum(1 for v in ks_results.values()
                           if v.get("distributions_similar"))
        self.log("Similar distributions: " + str(similar_dists) + "/" + str(len(ks_results)))

        self.log("Comparing correlations...")
        corr_sim = self._correlation_similarity(real_df, synthetic_df)
        self.log("Correlation diff: " + str(corr_sim["mean_correlation_diff"]))

        fidelity_score = self._compute_fidelity_score(col_stats, ks_results, corr_sim)
        self.log("Fidelity score: " + str(fidelity_score) + "%")

        fidelity_results = {
            "fidelity_score": fidelity_score,
            "column_stats_comparison": col_stats,
            "ks_test_results": ks_results,
            "correlation_similarity": corr_sim,
            "similar_columns": similar_cols,
            "similar_distributions": similar_dists,
            "grade": "A" if fidelity_score >= 80 else "B" if fidelity_score >= 60 else "C"
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_fidelity_results.json"
        with open(path, "w") as f:
            json.dump(fidelity_results, f, indent=2)

        state["fidelity_results"] = fidelity_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] StatisticalFidelityAgent: " +
            "score=" + str(fidelity_score) +
            " grade=" + fidelity_results["grade"]
        )

        self.log("=" * 50)
        self.log("STATISTICAL FIDELITY COMPLETE")
        self.log("Fidelity score : " + str(fidelity_score) + "%")
        self.log("Grade          : " + fidelity_results["grade"])
        self.log("Similar cols   : " + str(similar_cols))
        self.log("=" * 50)

        return state