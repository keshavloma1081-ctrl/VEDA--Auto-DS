"""
VEDA — Autonomous Data Science System
agents/timeseries/prophet_agent.py — Prophet Agent

Facebook Prophet forecasting:
- Trend + seasonality decomposition
- Holiday effects
- Uncertainty intervals
- Future forecasting
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from veda.core.base_agent import BaseAgent


class ProphetAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="ProphetAgent",
            domain="timeseries",
            version="1.0.0"
        )

    def _load_ts_data(self, state: dict) -> pd.DataFrame:
        ts_data = state.get("ts_data")
        if ts_data is not None:
            if isinstance(ts_data, pd.DataFrame):
                return ts_data
            return pd.DataFrame(ts_data)
        ts_eda = state.get("ts_eda", {})
        records = ts_eda.get("ts_data", [])
        if records:
            return pd.DataFrame(records)
        return None

    def run(self, state: dict) -> dict:
        self.log("Loading time series data...")
        df = self._load_ts_data(state)

        if df is None:
            self.log("No time series data found", level="WARN")
            return state

        df["ds"] = pd.to_datetime(df["ds"])
        self.log("Series length: " + str(len(df)))

        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size][["ds", "y"]].copy()
        test_df = df.iloc[train_size:][["ds", "y"]].copy()

        self.log("Fitting Prophet model...")
        try:
            from prophet import Prophet

            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(train_df)

            future = model.make_future_dataframe(periods=len(test_df) + 30)
            forecast = model.predict(future)

            test_forecast = forecast.iloc[train_size:train_size + len(test_df)]
            y_pred = test_forecast["yhat"].values
            y_true = test_df["y"].values

            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
            mape = float(np.mean(np.abs((y_true - y_pred) /
                                       np.maximum(np.abs(y_true), 1e-6))) * 100)

            future_30 = forecast.tail(30)
            components = {
                "trend": round(float(forecast["trend"].mean()), 4),
                "weekly_seasonality": "detected" if "weekly" in forecast.columns else "not available",
                "yearly_seasonality": "detected" if "yearly" in forecast.columns else "not available"
            }

            prophet_results = {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4),
                "components": components,
                "forecast_30d": [round(float(v), 4) for v in future_30["yhat"].values],
                "forecast_lower": [round(float(v), 4) for v in future_30["yhat_lower"].values],
                "forecast_upper": [round(float(v), 4) for v in future_30["yhat_upper"].values],
                "train_size": len(train_df),
                "test_size": len(test_df)
            }

        except Exception as e:
            self.log("Prophet failed: " + str(e), level="WARN")
            prophet_results = {"error": str(e)}

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_prophet_results.json"
        with open(path, "w") as f:
            json.dump(prophet_results, f, indent=2)

        state["prophet_results"] = prophet_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ProphetAgent: " +
            "MAE=" + str(prophet_results.get("mae", "N/A")) +
            " RMSE=" + str(prophet_results.get("rmse", "N/A"))
        )

        self.log("=" * 50)
        self.log("PROPHET COMPLETE")
        self.log("MAE  : " + str(prophet_results.get("mae", "N/A")))
        self.log("RMSE : " + str(prophet_results.get("rmse", "N/A")))
        self.log("MAPE : " + str(prophet_results.get("mape", "N/A")) + "%")
        self.log("=" * 50)

        return state