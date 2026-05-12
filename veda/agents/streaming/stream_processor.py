"""
VEDA — Autonomous Data Science System
agents/streaming/stream_processor.py — Stream Processor Agent

Real-time stream processing:
- Tumbling windows
- Sliding windows
- Aggregations
- Feature extraction
- Late event handling
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from veda.core.base_agent import BaseAgent


class StreamProcessorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="StreamProcessorAgent",
            domain="streaming",
            version="1.0.0"
        )

    def _tumbling_window(self, df: pd.DataFrame,
                          window_size: int = 60,
                          agg_cols: list = None) -> pd.DataFrame:
        """Tumbling window aggregations."""
        if "timestamp" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        numeric_cols = agg_cols or df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        df["window"] = df["timestamp"].dt.floor(str(window_size) + "s")

        agg_dict = {col: ["mean", "sum", "count", "std"] for col in numeric_cols}
        windowed = df.groupby("window").agg(agg_dict)
        windowed.columns = ["_".join(col) for col in windowed.columns]
        windowed = windowed.reset_index()

        self.log("Tumbling windows: " + str(len(windowed)) + " windows of " +
                str(window_size) + "s")
        return windowed

    def _sliding_window(self, df: pd.DataFrame,
                         window_size: int = 60,
                         slide_size: int = 30) -> list:
        """Sliding window processing."""
        if "timestamp" not in df.columns:
            return []

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        if df.empty:
            return []

        start = df["timestamp"].min()
        end = df["timestamp"].max()

        windows = []
        current = start
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        while current <= end:
            window_end = current + timedelta(seconds=window_size)
            mask = (df["timestamp"] >= current) & (df["timestamp"] < window_end)
            window_df = df[mask]

            if len(window_df) > 0:
                stats = {
                    "window_start": current.isoformat(),
                    "window_end": window_end.isoformat(),
                    "count": len(window_df),
                }
                for col in numeric_cols[:3]:
                    stats[col + "_mean"] = round(float(window_df[col].mean()), 4)
                    stats[col + "_std"] = round(float(window_df[col].std()), 4)
                windows.append(stats)

            current += timedelta(seconds=slide_size)

        self.log("Sliding windows: " + str(len(windows)) + " windows")
        return windows

    def _extract_stream_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from stream data."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["minute"] = df["timestamp"].dt.minute
            df["is_peak_hour"] = df["hour"].apply(lambda x: 1 if 9 <= x <= 17 else 0)

        for col in numeric_cols[:3]:
            df[col + "_rolling_mean"] = df[col].rolling(10, min_periods=1).mean()
            df[col + "_rolling_std"] = df[col].rolling(10, min_periods=1).std().fillna(0)

        if "source" in df.columns:
            source_dummies = pd.get_dummies(df["source"], prefix="source")
            df = pd.concat([df, source_dummies], axis=1)

        self.log("Stream features extracted: " + str(len(df.columns)) + " columns")
        return df

    def _compute_aggregations(self, df: pd.DataFrame) -> dict:
        """Compute global stream aggregations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        aggs = {}
        for col in numeric_cols[:5]:
            aggs[col] = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "sum": round(float(df[col].sum()), 4)
            }
        return aggs

    def run(self, state: dict) -> dict:
        stream_data = state.get("stream_data")
        if stream_data is None:
            d = "outputs"
            files = [f for f in os.listdir(d) if f.endswith("_stream_data.parquet")]
            if files:
                stream_data = pd.read_parquet(os.path.join(d, sorted(files)[-1]))
            else:
                self.log("No stream data found", level="WARN")
                return state

        if not isinstance(stream_data, pd.DataFrame):
            stream_data = pd.DataFrame(stream_data)

        self.log("Processing stream: " + str(stream_data.shape))

        agg_cols = ["value", "quantity", "latency_ms", "user_score"]
        agg_cols = [c for c in agg_cols if c in stream_data.columns]

        self.log("Running tumbling window (60s)...")
        tumbling = self._tumbling_window(stream_data, window_size=60, agg_cols=agg_cols)

        self.log("Running sliding window (60s, 30s slide)...")
        sliding = self._sliding_window(stream_data, window_size=60, slide_size=30)

        self.log("Extracting stream features...")
        enriched_df = self._extract_stream_features(stream_data)

        self.log("Computing aggregations...")
        aggregations = self._compute_aggregations(stream_data)

        processor_results = {
            "input_events": len(stream_data),
            "tumbling_windows": len(tumbling),
            "sliding_windows": len(sliding),
            "enriched_features": len(enriched_df.columns),
            "aggregations": aggregations,
            "sliding_sample": sliding[:3]
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        enriched_path = "outputs/" + run_id + "_stream_features.parquet"
        enriched_df.to_parquet(enriched_path, index=False)

        results_path = "outputs/" + run_id + "_processor_results.json"
        with open(results_path, "w") as f:
            json.dump(processor_results, f, indent=2, default=str)

        state["stream_processor"] = processor_results
        state["stream_features"] = enriched_df
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] StreamProcessorAgent: " +
            str(len(tumbling)) + " tumbling, " +
            str(len(sliding)) + " sliding windows"
        )

        self.log("=" * 50)
        self.log("STREAM PROCESSOR COMPLETE")
        self.log("Input events    : " + str(len(stream_data)))
        self.log("Tumbling windows: " + str(len(tumbling)))
        self.log("Sliding windows : " + str(len(sliding)))
        self.log("Features        : " + str(len(enriched_df.columns)))
        self.log("=" * 50)

        return state