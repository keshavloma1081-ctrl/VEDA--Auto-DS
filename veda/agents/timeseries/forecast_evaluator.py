"""
VEDA — Autonomous Data Science System
agents/timeseries/forecast_evaluator.py — Forecast Evaluator Agent

Compares all forecasting models:
- ARIMA vs Prophet vs LSTM
- Best model selection
- Forecast ensemble
- Business recommendations
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class ForecastEvaluatorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="ForecastEvaluatorAgent",
            domain="timeseries",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _compare_models(self, state: dict) -> dict:
        """Compare all forecasting models."""
        models = {}

        arima = state.get("arima_results", {})
        if arima and "mae" in arima:
            models["ARIMA"] = {
                "mae": arima["mae"],
                "rmse": arima["rmse"],
                "mape": arima["mape"],
                "order": arima.get("order")
            }

        prophet = state.get("prophet_results", {})
        if prophet and "mae" in prophet:
            models["Prophet"] = {
                "mae": prophet["mae"],
                "rmse": prophet["rmse"],
                "mape": prophet["mape"]
            }

        lstm = state.get("lstm_forecast", {})
        if lstm and "mae" in lstm:
            models["LSTM"] = {
                "mae": lstm["mae"],
                "rmse": lstm["rmse"],
                "mape": lstm["mape"]
            }

        return models

    def _select_best_model(self, models: dict) -> str:
        """Select best model by MAE."""
        if not models:
            return "N/A"
        return min(models, key=lambda k: models[k].get("mae", float("inf")))

    def _ensemble_forecast(self, state: dict) -> list:
        """Simple average ensemble of all forecasts."""
        forecasts = []

        arima = state.get("arima_results", {}).get("forecast_30d", [])
        prophet = state.get("prophet_results", {}).get("forecast_30d", [])
        lstm = state.get("lstm_forecast", {}).get("forecast_30d", [])

        available = [f for f in [arima, prophet, lstm] if f]
        if not available:
            return []

        min_len = min(len(f) for f in available)
        for i in range(min_len):
            avg = sum(f[i] for f in available) / len(available)
            forecasts.append(round(float(avg), 4))

        return forecasts

    def _generate_forecast_report(self, models: dict,
                                   best: str, ensemble: list) -> str:
        """Generate LLM forecast report."""
        prompt = """You are a forecasting expert. Summarize these model results.

Models compared:
""" + json.dumps(models, indent=2) + """

Best model: """ + best + """
Ensemble forecast (30 days): """ + str(ensemble[:5]) + "..." + """

Write 3 sentences:
1. Which model performed best and why
2. What the forecast suggests about the trend
3. Business recommendation based on the forecast"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except:
            return "Forecast evaluation complete. " + best + " performed best."

    def run(self, state: dict) -> dict:
        self.log("Comparing forecasting models...")
        models = self._compare_models(state)

        if not models:
            self.log("No model results found", level="WARN")
            return state

        self.log("Models compared: " + str(list(models.keys())))

        for name, metrics in models.items():
            self.log(name + " MAE=" + str(metrics.get("mae")) +
                    " RMSE=" + str(metrics.get("rmse")) +
                    " MAPE=" + str(metrics.get("mape")) + "%")

        best = self._select_best_model(models)
        self.log("Best model: " + best)

        self.log("Computing ensemble forecast...")
        ensemble = self._ensemble_forecast(state)
        self.log("Ensemble forecast (first 5): " + str(ensemble[:5]))

        self.log("Generating forecast report...")
        report = self._generate_forecast_report(models, best, ensemble)

        evaluation = {
            "models": models,
            "best_model": best,
            "ensemble_forecast_30d": ensemble,
            "report": report,
            "ranking": sorted(models.keys(),
                            key=lambda k: models[k].get("mae", float("inf")))
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_forecast_evaluation.json"
        with open(path, "w") as f:
            json.dump(evaluation, f, indent=2)

        state["forecast_evaluation"] = evaluation
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ForecastEvaluatorAgent: " +
            "best=" + best +
            " models=" + str(list(models.keys()))
        )

        self.log("=" * 50)
        self.log("FORECAST EVALUATION COMPLETE")
        self.log("Best model : " + best)
        self.log("Ranking    : " + str(evaluation["ranking"]))
        self.log("Report     : " + report[:100] + "...")
        self.log("=" * 50)

        return state