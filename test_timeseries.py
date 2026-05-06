"""Test VEDA Time Series Agents"""
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

state = {
    "run_id": run_id,
    "goal": "forecast daily sales for next 30 days",
    "planner_decision_log": []
}

print("="*50)
print("Testing Time Series Agents")
print("="*50)

print("\n[1/5] Time Series EDA Agent...")
from veda.agents.timeseries.ts_eda import TimeSeriesEDAAgent
agent1 = TimeSeriesEDAAgent()
state = agent1.execute(state)
eda = state.get("ts_eda", {})
print("Series length : " + str(eda.get("series_length")))
print("Trend         : " + str(eda.get("trend", {}).get("trend_direction")))
print("Has seasonality: " + str(eda.get("seasonality", {}).get("has_seasonality")))

print("\n[2/5] ARIMA Agent...")
from veda.agents.timeseries.arima_agent import ARIMAAgent
agent2 = ARIMAAgent()
state = agent2.execute(state)
arima = state.get("arima_results", {})
print("Order : " + str(arima.get("order")))
print("MAE   : " + str(arima.get("mae")))
print("RMSE  : " + str(arima.get("rmse")))

print("\n[3/5] Prophet Agent...")
from veda.agents.timeseries.prophet_agent import ProphetAgent
agent3 = ProphetAgent()
state = agent3.execute(state)
prophet = state.get("prophet_results", {})
print("MAE  : " + str(prophet.get("mae")))
print("RMSE : " + str(prophet.get("rmse")))

print("\n[4/5] LSTM Forecaster Agent...")
from veda.agents.timeseries.lstm_forecaster import LSTMForecasterAgent
agent4 = LSTMForecasterAgent()
state = agent4.execute(state)
lstm = state.get("lstm_forecast", {})
print("MAE  : " + str(lstm.get("mae")))
print("RMSE : " + str(lstm.get("rmse")))

print("\n[5/5] Forecast Evaluator Agent...")
from veda.agents.timeseries.forecast_evaluator import ForecastEvaluatorAgent
agent5 = ForecastEvaluatorAgent()
state = agent5.execute(state)
evaluation = state.get("forecast_evaluation", {})
print("Best model : " + str(evaluation.get("best_model")))
print("Ranking    : " + str(evaluation.get("ranking")))

print("\n" + "="*50)
print("TIME SERIES PIPELINE COMPLETE")
print("="*50)
for log in state.get("planner_decision_log", []):
    print("  " + log)