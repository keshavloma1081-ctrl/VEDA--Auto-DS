"""
VEDA — Autonomous Data Science System
agents/timeseries/arima_agent.py — ARIMA Agent

Auto ARIMA forecasting:
- Auto parameter selection
- Model fitting
- Forecasting
- Residual analysis
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from veda.core.base_agent import BaseAgent


class ARIMAAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="ARIMAAgent",
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

    def _auto_arima(self, series: pd.Series) -> dict:
        """Fit ARIMA with auto parameter selection."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller

            p_value = adfuller(series.dropna())[1]
            d = 0 if p_value < 0.05 else 1

            best_aic = np.inf
            best_order = (1, d, 1)
            best_model = None

            for p in range(0, 3):
                for q in range(0, 3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model = fitted
                    except:
                        continue

            return best_model, best_order, best_aic

        except Exception as e:
            self.log("ARIMA fitting failed: " + str(e), level="WARN")
            return None, (1, 1, 1), np.inf

    def _compute_residual_stats(self, residuals: np.ndarray) -> dict:
        """Compute residual statistics."""
        return {
            "mean": round(float(residuals.mean()), 6),
            "std": round(float(residuals.std()), 6),
            "max_abs": round(float(np.abs(residuals).max()), 4)
        }

    def run(self, state: dict) -> dict:
        self.log("Loading time series data...")
        df = self._load_ts_data(state)

        if df is None:
            self.log("No time series data found", level="WARN")
            return state

        df["ds"] = pd.to_datetime(df["ds"])
        series = df.set_index("ds")["y"]
        self.log("Series length: " + str(len(series)))

        train_size = int(len(series) * 0.8)
        train = series.iloc[:train_size]
        test = series.iloc[train_size:]
        self.log("Train: " + str(len(train)) + " Test: " + str(len(test)))

        self.log("Fitting Auto-ARIMA...")
        model, order, aic = self._auto_arima(train)
        self.log("Best order: " + str(order) + " AIC: " + str(round(aic, 2)))

        if model is not None:
            forecast = model.forecast(steps=len(test))
            forecast_values = forecast.values if hasattr(forecast, "values") else np.array(forecast)

            mae = float(np.mean(np.abs(test.values - forecast_values)))
            rmse = float(np.sqrt(np.mean((test.values - forecast_values)**2)))
            mape = float(np.mean(np.abs((test.values - forecast_values) /
                                       np.maximum(np.abs(test.values), 1e-6))) * 100)

            future_forecast = model.forecast(steps=30)
            future_values = future_forecast.values if hasattr(future_forecast, "values") else np.array(future_forecast)

            residuals = model.resid
            residual_stats = self._compute_residual_stats(residuals.values)

            arima_results = {
                "order": list(order),
                "aic": round(float(aic), 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4),
                "residual_stats": residual_stats,
                "forecast_30d": [round(float(v), 4) for v in future_values],
                "train_size": len(train),
                "test_size": len(test)
            }
        else:
            arima_results = {"error": "ARIMA fitting failed", "order": list(order)}

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_arima_results.json"
        with open(path, "w") as f:
            json.dump(arima_results, f, indent=2)

        state["arima_results"] = arima_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ARIMAAgent: " +
            "order=" + str(order) +
            " MAE=" + str(arima_results.get("mae", "N/A")) +
            " RMSE=" + str(arima_results.get("rmse", "N/A"))
        )

        self.log("=" * 50)
        self.log("ARIMA COMPLETE")
        self.log("Order : " + str(order))
        self.log("AIC   : " + str(round(float(aic), 2)))
        self.log("MAE   : " + str(arima_results.get("mae", "N/A")))
        self.log("RMSE  : " + str(arima_results.get("rmse", "N/A")))
        self.log("MAPE  : " + str(arima_results.get("mape", "N/A")) + "%")
        self.log("=" * 50)

        return state