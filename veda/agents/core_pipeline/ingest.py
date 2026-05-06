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
        source_type = self._detect_source(path)
        self.log("Source type detected: " + source_type)
        if source_type == "csv":
            df = pd.read_csv(path)
        elif source_type == "parquet":
            df = pd.read_parquet(path)
        elif source_type == "json":
            df = pd.read_json(path)
        elif source_type == "excel":
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type: " + source_type)
        return df

    def _profile_data(self, df: pd.DataFrame, target_column: str = None) -> dict:
        row_count, col_count = df.shape
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        null_counts = df.isnull().sum().to_dict()
        all_columns = list(df.columns)

        if target_column and target_column in all_columns:
            feature_columns = [c for c in all_columns if c != target_column]
        else:
            feature_columns = all_columns
            target_column = None

        class_balance = None
        has_imbalance = False
        if target_column and target_column in df.columns:
            if df[target_column].dtype == "object" or df[target_column].nunique() < 20:
                counts = df[target_column].value_counts()
                class_balance = counts.to_dict()
                if len(counts) >= 2:
                    ratio = counts.iloc[-1] / counts.iloc[0]
                    has_imbalance = ratio < 0.2

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
        dataset_path = state.get("dataset_path", "")
        goal = state.get("goal", "")

        target_column = None
        if "target:" in goal.lower():
            try:
                target_column = goal.lower().split("target:")[1].strip().split()[0]
            except:
                pass

        self.log("Loading data from: " + dataset_path)

        # Check file size
        file_size_gb = os.path.getsize(dataset_path) / (1024 ** 3)
        file_size_mb = os.path.getsize(dataset_path) / (1024 ** 2)
        self.log("File size: " + str(round(file_size_mb, 1)) + " MB")

        if file_size_gb > 0.5:
            # Large file — load sample only
            self.log("Large file detected — sampling 500K rows for training...")
            df = pd.read_csv(dataset_path, nrows=500_000)
            self.log("Sampled 500,000 rows from large dataset")
        else:
            df = self._load_data(dataset_path)

        self.log("Loaded " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " columns")

        profile = self._profile_data(df, target_column)
        state["data_profile"] = profile

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_path = "outputs/" + run_id + "_data.parquet"
        df.to_parquet(output_path, index=False)
        self.log("Data saved to: " + output_path)

        state["feature_list"] = profile["feature_columns"]
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] DataIngest: loaded " +
            str(profile["row_count"]) + " rows, " +
            str(profile["col_count"]) + " columns, target=" +
            str(profile["target_column"])
        )

        self.log("Rows      : " + str(profile["row_count"]))
        self.log("Columns   : " + str(profile["col_count"]))
        self.log("Target    : " + str(profile["target_column"]))
        self.log("Nulls     : " + str(profile["null_counts"]))
        self.log("Imbalance : " + str(profile["has_imbalance"]))

        return state