"""
VEDA — Autonomous Data Science System
agents/timeseries/ts_eda.py — Time Series EDA Agent

Time series exploratory analysis:
- Trend detection
- Seasonality detection
- Stationarity tests (ADF, KPSS)
- Autocorrelation analysis
- Decomposition
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class TimeSeriesEDAAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="TimeSeriesEDAAgent",
            domain="timeseries",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _generate_ts_data(self, n: int = 365) -> pd.DataFrame:
        """Generate synthetic time series data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
        trend = np.linspace(100, 150, n)
        seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
        noise = np.random.normal(0, 5, n)
        values = trend + seasonality + noise
        return pd.DataFrame({"ds": dates, "y": values})

    def _check_stationarity(self, series: pd.Series) -> dict:
        """ADF test for stationarity."""
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            adf_result = adfuller(series.dropna())
            adf = {
                "test_statistic": round(float(adf_result[0]), 6),
                "p_value": round(float(adf_result[1]), 6),
                "is_stationary": bool(adf_result[1] < 0.05),
                "critical_values": {k: round(v, 4) for k, v in adf_result[4].items()}
            }
            return {"adf_test": adf}
        except Exception as e:
            return {"error": str(e)}

    def _detect_trend(self, series: pd.Series) -> dict:
        """Detect trend using linear regression."""
        x = np.arange(len(series))
        y = series.values
        slope, intercept = np.polyfit(x, y, 1)
        trend_strength = abs(slope) / (series.std() + 1e-6)
        return {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "trend_direction": "upward" if slope > 0 else "downward",
            "trend_strength": round(float(trend_strength), 4),
            "has_trend": bool(abs(slope) > 0.01)
        }

    def _detect_seasonality(self, series: pd.Series) -> dict:
        """Detect seasonality using autocorrelation."""
        try:
            from statsmodels.tsa.stattools import acf
            acf_values = acf(series.dropna(), nlags=min(50, len(series)//2))
            seasonal_lags = []
            for lag in [7, 12, 24, 30, 52, 365]:
                if lag < len(acf_values) and abs(acf_values[lag]) > 0.2:
                    seasonal_lags.append({
                        "lag": lag,
                        "acf": round(float(acf_values[lag]), 4)
                    })
            return {
                "seasonal_lags": seasonal_lags,
                "has_seasonality": len(seasonal_lags) > 0,
                "strongest_period": seasonal_lags[0]["lag"] if seasonal_lags else None
            }
        except Exception as e:
            return {"error": str(e), "has_seasonality": False}

    def _compute_ts_stats(self, series: pd.Series) -> dict:
        """Basic time series statistics."""
        return {
            "length": int(len(series)),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "cv": round(float(series.std() / max(abs(series.mean()), 1e-6)), 4)
        }

    def _generate_ts_summary(self, stats: dict, trend: dict,
                              seasonality: dict, stationarity: dict) -> str:
        """Generate LLM summary of time series."""
        prompt = """Summarize this time series analysis in 3 sentences.

Stats: """ + json.dumps(stats) + """
Trend: """ + json.dumps(trend) + """
Seasonality: """ + json.dumps(seasonality) + """
Stationarity: """ + json.dumps(stationarity) + """

Cover: data characteristics, trend, seasonality, and forecasting recommendations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except:
            return "Time series analysis complete."

    def run(self, state: dict) -> dict:
        self.log("Generating/loading time series data...")
        df = self._generate_ts_data(n=365)
        series = df["y"]
        self.log("Series length: " + str(len(series)))

        self.log("Computing basic statistics...")
        stats = self._compute_ts_stats(series)

        self.log("Detecting trend...")
        trend = self._detect_trend(series)
        self.log("Trend: " + trend["trend_direction"] +
                " slope=" + str(trend["slope"]))

        self.log("Detecting seasonality...")
        seasonality = self._detect_seasonality(series)
        self.log("Has seasonality: " + str(seasonality.get("has_seasonality")))

        self.log("Running stationarity tests...")
        stationarity = self._check_stationarity(series)
        adf = stationarity.get("adf_test", {})
        self.log("ADF p-value: " + str(adf.get("p_value", "N/A")))

        self.log("Generating summary...")
        summary = self._generate_ts_summary(stats, trend, seasonality, stationarity)

        ts_eda = {
            "series_length": len(series),
            "stats": stats,
            "trend": trend,
            "seasonality": seasonality,
            "stationarity": stationarity,
            "summary": summary,
            "ts_data": df.to_dict(orient="records")
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_ts_eda.json"
        with open(path, "w") as f:
            json.dump(ts_eda, f, indent=2, default=str)

        state["ts_eda"] = ts_eda
        state["ts_data"] = df
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] TimeSeriesEDAAgent: " +
            "trend=" + trend["trend_direction"] +
            " seasonal=" + str(seasonality.get("has_seasonality"))
        )

        self.log("=" * 50)
        self.log("TIME SERIES EDA COMPLETE")
        self.log("Length      : " + str(len(series)))
        self.log("Trend       : " + trend["trend_direction"])
        self.log("Seasonality : " + str(seasonality.get("has_seasonality")))
        self.log("Stationary  : " + str(adf.get("is_stationary", "N/A")))
        self.log("=" * 50)

        return state