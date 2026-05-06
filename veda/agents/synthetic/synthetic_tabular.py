"""
VEDA — Autonomous Data Science System
agents/synthetic/synthetic_tabular.py — Synthetic Tabular Agent

Generates synthetic tabular data using:
- GaussianCopula (SDV)
- CTGAN (SDV)
- Statistical sampling fallback
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent


class SyntheticTabularAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="SyntheticTabularAgent",
            domain="synthetic",
            version="1.0.0"
        )

    def _load_data(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _gaussian_copula(self, df: pd.DataFrame,
                          n_rows: int = 500) -> pd.DataFrame:
        """Generate synthetic data using GaussianCopula."""
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            from sdv.metadata import SingleTableMetadata

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)

            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer.fit(df)
            synthetic = synthesizer.sample(num_rows=n_rows)
            self.log("GaussianCopula generated " + str(len(synthetic)) + " rows")
            return synthetic, "GaussianCopula"
        except Exception as e:
            self.log("GaussianCopula failed: " + str(e), level="WARN")
            return None, None

    def _statistical_sampling(self, df: pd.DataFrame,
                               n_rows: int = 500) -> pd.DataFrame:
        """Statistical sampling fallback."""
        np.random.seed(42)
        synthetic_data = {}

        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                mean = df[col].mean()
                std = df[col].std()
                synthetic_data[col] = np.random.normal(mean, std + 1e-6, n_rows)
                if df[col].dtype in [np.int64, np.int32]:
                    synthetic_data[col] = np.round(synthetic_data[col]).astype(int)
            elif df[col].dtype == object or str(df[col].dtype) == "category":
                value_counts = df[col].value_counts(normalize=True)
                synthetic_data[col] = np.random.choice(
                    value_counts.index,
                    size=n_rows,
                    p=value_counts.values
                )
            else:
                synthetic_data[col] = df[col].sample(n_rows, replace=True).values

        return pd.DataFrame(synthetic_data), "StatisticalSampling"

    def _compute_basic_stats(self, df: pd.DataFrame) -> dict:
        """Compute basic statistics for comparison."""
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4)
            }
        return stats

    def run(self, state: dict) -> dict:
        self.log("Loading real data...")
        df = self._load_data(state)

        if df is None:
            self.log("No data found", level="WARN")
            return state

        self.log("Real data shape: " + str(df.shape))
        n_synthetic = min(len(df), 500)

        # Try GaussianCopula first
        self.log("Attempting GaussianCopula synthesis...")
        synthetic_df, method = self._gaussian_copula(df, n_rows=n_synthetic)

        if synthetic_df is None:
            self.log("Falling back to statistical sampling...")
            synthetic_df, method = self._statistical_sampling(df, n_rows=n_synthetic)

        self.log("Method used: " + method)
        self.log("Synthetic data shape: " + str(synthetic_df.shape))

        real_stats = self._compute_basic_stats(df)
        synthetic_stats = self._compute_basic_stats(synthetic_df)

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        synthetic_path = "outputs/" + run_id + "_synthetic_data.parquet"
        synthetic_df.to_parquet(synthetic_path, index=False)

        results = {
            "method": method,
            "real_rows": len(df),
            "synthetic_rows": len(synthetic_df),
            "columns": list(df.columns),
            "real_stats": real_stats,
            "synthetic_stats": synthetic_stats,
            "synthetic_path": synthetic_path
        }

        path = "outputs/" + run_id + "_synthetic_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        state["synthetic_results"] = results
        state["synthetic_df"] = synthetic_df
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] SyntheticTabularAgent: " +
            method + " " + str(len(synthetic_df)) + " rows generated"
        )

        self.log("=" * 50)
        self.log("SYNTHETIC DATA COMPLETE")
        self.log("Method    : " + method)
        self.log("Real rows : " + str(len(df)))
        self.log("Synth rows: " + str(len(synthetic_df)))
        self.log("=" * 50)

        return state