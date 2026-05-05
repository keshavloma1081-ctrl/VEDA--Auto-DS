"""
VEDA — Autonomous Data Science System
agents/core_pipeline/ingest.py — Data Ingest Agent

Entry point for all data. Accepts CSV, Parquet, JSON, Excel.
Loads data, infers schema, profiles it, and writes
a clean typed DataFrame summary to VEDAState.
"""

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from veda.core.base_agent import BaseAgent

load_dotenv()


class IngestAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="DataIngest",
            domain="data_engineering",
            version="1.0.0"
        )

    def _detect_source(self, path: str) -> str:
        """Detect file type from extension."""
        ext = path.lower().split(".")[-1]
        mapping = {
            "csv": "csv",
            "parquet": "parquet",
            "json": "json",
            "xlsx": "excel",
            "xls": "excel"
        }
        return mapping.get(ext, "csv")

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from file path."""
        source_type = self._detect_source(path)

        self.log(f"Source type detected: {source_type}")

        if source_type == "csv":
            df = pd.read_csv(path)
        elif source_type == "parquet":
            df = pd.read_parquet(path)
        elif source_type == "json":
            df = pd.read_json(path)
        elif source_type == "excel":
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {source_type}")

        return df

    def _profile_data(self, df: pd.DataFrame, target_column: str = None) -> dict:
        """Generate a lightweight data profile."""

        # Basic shape
        row_count, col_count = df.shape

        # Dtypes
        dtypes = {col: str(df[col].dtype) for col in df.columns}

        # Null counts
        null_counts = df.isnull().sum().to_dict()

        # Feature columns (everything except target)
        all_columns = list(df.columns)
        if target_column and target_column in all_columns:
            feature_columns = [c for c in all_columns if c != target_column]
        else:
            feature_columns = all_columns
            target_column = None

        # Class balance (if target exists and is categorical)
        class_balance = None
        has_imbalance = False
        if target_column and target_column in df.columns:
            if df[target_column].dtype == "object" or df[target_column].nunique() < 20:
                counts = df[target_column].value_counts()
                class_balance = counts.to_dict()
                # Flag imbalance if minority class < 20% of majority
                if len(counts) >= 2:
                    ratio = counts.iloc[-1] / counts.iloc[0]
                    has_imbalance = ratio < 0.2

        # Duplicate check
        duplicate_count = df.duplicated().sum()

        return {
            "row_count": row_count,
            "col_count": col_count,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "null_counts": null_counts,
            "dtypes": dtypes,
            "class_balance": class_balance,
            "has_imbalance": has_imbalance,
            "duplicate_count": int(duplicate_count),
            "has_leakage_risk": False,
            "eda_summary": None
        }

    def run(self, state: dict) -> dict:
        """
        Main ingest logic:
        1. Load data from dataset_path
        2. Profile it
        3. Save profile to state
        4. Save data to outputs folder
        """

        dataset_path = state.get("dataset_path", "")
        execution_plan = state.get("execution_plan", {})

        # Get target column from goal if possible
        # For now we ask user to include it in goal as "target: column_name"
        goal = state.get("goal", "")
        target_column = None
        if "target:" in goal.lower():
            try:
                target_column = goal.lower().split("target:")[1].strip().split()[0]
            except:
                pass

        self.log(f"Loading data from: {dataset_path}")

        # Step 1 — load
        df = self._load_data(dataset_path)
        self.log(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

        # Step 2 — profile
        self.log("Profiling data...")
        profile = self._profile_data(df, target_column)

        # Step 3 — save profile to state
        state["data_profile"] = profile

        # Step 4 — save clean data to outputs folder
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_path = f"outputs/{run_id}_data.parquet"
        df.to_parquet(output_path, index=False)
        self.log(f"Data saved to: {output_path}")

        # Step 5 — update state
        state["feature_list"] = profile["feature_columns"]
        state["planner_decision_log"] = state.get("planner_decision_log", [])
        state["planner_decision_log"].append(
            f"[{datetime.now().isoformat()}] DataIngest: loaded {profile['row_count']} rows, "
            f"{profile['col_count']} columns, target={profile['target_column']}"
        )

        # Print summary
        self.log(f"Rows: {profile['row_count']}")
        self.log(f"Columns: {profile['col_count']}")
        self.log(f"Target column: {profile['target_column']}")
        self.log(f"Null counts: {profile['null_counts']}")
        self.log(f"Class imbalance: {profile['has_imbalance']}")

        return state