"""
VEDA — Autonomous Data Science System
agents/core_pipeline/cleaning.py — Data Cleaning Agent

Reads EDA findings and fixes data quality issues:
- Null imputation
- Duplicate removal
- Outlier treatment
- Type coercion
- Leakage detection
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from veda.core.base_agent import BaseAgent

load_dotenv()


class CleaningAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="CleaningAgent",
            domain="data_engineering",
            version="1.0.0"
        )

    def _load_data(self, state: dict) -> pd.DataFrame:
        """Load the saved parquet file from outputs."""
        outputs_dir = "outputs"
        files = [f for f in os.listdir(outputs_dir) if f.endswith("_data.parquet")]
        if not files:
            return pd.read_csv(state.get("dataset_path", ""))
        latest = sorted(files)[-1]
        return pd.read_parquet(os.path.join(outputs_dir, latest))

    def _impute_nulls(self, df: pd.DataFrame, target_col: str = None) -> tuple:
        """Impute missing values column by column."""
        changes = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Remove target from imputation
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        for col in numeric_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_pct = null_count / len(df)
                if null_pct > 0.6:
                    df = df.drop(columns=[col])
                    changes.append(f"DROPPED {col} — {null_pct:.0%} nulls")
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    changes.append(f"IMPUTED {col} with median={median_val:.2f} ({null_count} nulls)")

        for col in cat_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_pct = null_count / len(df)
                if null_pct > 0.6:
                    df = df.drop(columns=[col])
                    changes.append(f"DROPPED {col} — {null_pct:.0%} nulls")
                else:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    changes.append(f"IMPUTED {col} with mode='{mode_val}' ({null_count} nulls)")

        return df, changes

    def _remove_duplicates(self, df: pd.DataFrame) -> tuple:
        """Remove exact duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        changes = []
        if removed > 0:
            changes.append(f"REMOVED {removed} duplicate rows")
        return df, changes

    def _treat_outliers(self, df: pd.DataFrame, target_col: str = None) -> tuple:
        """Winsorise outliers using IQR method."""
        changes = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            if outlier_count > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                changes.append(
                    f"WINSORISED {col} — {outlier_count} outliers clipped to [{lower:.2f}, {upper:.2f}]"
                )

        return df, changes

    def _detect_leakage(self, df: pd.DataFrame, target_col: str = None) -> list:
        """Flag features suspiciously correlated with target."""
        flags = []
        if not target_col or target_col not in df.columns:
            return flags

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col not in numeric_cols:
            return flags

        for col in numeric_cols:
            if col == target_col:
                continue
            corr = abs(df[col].corr(df[target_col]))
            if corr > 0.95:
                flags.append(f"LEAKAGE RISK: {col} has {corr:.3f} correlation with target")

        return flags

    def run(self, state: dict) -> dict:
        """
        Main cleaning logic:
        1. Load data
        2. Remove duplicates
        3. Impute nulls
        4. Treat outliers
        5. Detect leakage
        6. Save cleaned data
        """

        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading dataset for cleaning...")
        df = self._load_data(state)
        original_shape = df.shape
        self.log(f"Original shape: {original_shape}")

        all_changes = []

        # Step 1 — remove duplicates
        self.log("Removing duplicates...")
        df, dup_changes = self._remove_duplicates(df)
        all_changes.extend(dup_changes)

        # Step 2 — impute nulls
        self.log("Imputing null values...")
        df, null_changes = self._impute_nulls(df, target_col)
        all_changes.extend(null_changes)

        # Step 3 — treat outliers
        self.log("Treating outliers with winsorisation...")
        df, outlier_changes = self._treat_outliers(df, target_col)
        all_changes.extend(outlier_changes)

        # Step 4 — detect leakage
        self.log("Checking for data leakage...")
        leakage_flags = self._detect_leakage(df, target_col)
        if leakage_flags:
            for flag in leakage_flags:
                self.log(flag, level="WARN")
        else:
            self.log("No leakage detected")

        # Step 5 — save cleaned data
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        cleaned_path = f"outputs/{run_id}_cleaned.parquet"
        df.to_parquet(cleaned_path, index=False)
        self.log(f"Cleaned data saved to: {cleaned_path}")

        # Step 6 — update state
        state["cleaning_diff"] = all_changes
        if state.get("data_profile"):
            state["data_profile"]["has_leakage_risk"] = len(leakage_flags) > 0
            state["data_profile"]["row_count"] = len(df)
            state["data_profile"]["col_count"] = len(df.columns)

        state["planner_decision_log"] = state.get("planner_decision_log", [])
        state["planner_decision_log"].append(
            f"[{datetime.now().isoformat()}] CleaningAgent: {len(all_changes)} changes made"
        )

        # Print summary
        self.log("=" * 50)
        self.log(f"CLEANING COMPLETE")
        self.log(f"Original : {original_shape[0]} rows x {original_shape[1]} cols")
        self.log(f"Cleaned  : {df.shape[0]} rows x {df.shape[1]} cols")
        self.log(f"Changes  : {len(all_changes)}")
        for change in all_changes:
            self.log(f"  → {change}")
        self.log("=" * 50)

        return state