"""
VEDA — Autonomous Data Science System
agents/timeseries/lstm_forecaster.py — LSTM Forecaster Agent

LSTM-based time series forecasting:
- Sequence preparation
- LSTM model training
- Multi-step forecasting
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from veda.core.base_agent import BaseAgent


class LSTMForecastNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMForecasterAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="LSTMForecasterAgent",
            domain="timeseries",
            version="1.0.0"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 30

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

    def _prepare_sequences(self, series: np.ndarray,
                            seq_len: int) -> tuple:
        X, y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i:i+seq_len])
            y.append(series[i+seq_len])
        return np.array(X), np.array(y)

    def _normalize(self, series: np.ndarray) -> tuple:
        mean = series.mean()
        std = series.std() + 1e-6
        return (series - mean) / std, mean, std

    def run(self, state: dict) -> dict:
        self.log("Loading time series data...")
        df = self._load_ts_data(state)

        if df is None:
            self.log("No time series data found", level="WARN")
            return state

        df["ds"] = pd.to_datetime(df["ds"])
        values = df["y"].values.astype(np.float32)
        self.log("Series length: " + str(len(values)))

        norm_values, mean, std = self._normalize(values)

        train_size = int(len(norm_values) * 0.8)
        train_vals = norm_values[:train_size]
        test_vals = norm_values[train_size:]

        X_train, y_train = self._prepare_sequences(train_vals, self.seq_len)
        X_test, y_test = self._prepare_sequences(
            np.concatenate([train_vals[-self.seq_len:], test_vals]), self.seq_len
        )

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).unsqueeze(2),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = LSTMForecastNet(
            input_size=1, hidden_size=64, num_layers=2
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        self.log("Training LSTM forecaster for 30 epochs...")
        for epoch in range(30):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0:
                self.log("Epoch " + str(epoch) + " loss=" +
                        str(round(epoch_loss / len(train_loader), 6)))

        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).unsqueeze(2).to(self.device)
            y_pred_norm = model(X_test_t).cpu().numpy().flatten()

        y_pred = y_pred_norm * std + mean
        y_true = y_test * std + mean

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        mape = float(np.mean(np.abs((y_true - y_pred) /
                                   np.maximum(np.abs(y_true), 1e-6))) * 100)

        last_seq = torch.FloatTensor(norm_values[-self.seq_len:]).unsqueeze(0).unsqueeze(2).to(self.device)
        future_preds = []
        current_seq = last_seq.clone()
        with torch.no_grad():
            for _ in range(30):
                pred = model(current_seq)
                future_preds.append(float(pred.item()))
                new_val = pred.unsqueeze(1)
                current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)

        future_denorm = [round(float(v * std + mean), 4) for v in future_preds]

        lstm_results = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "seq_len": self.seq_len,
            "hidden_size": 64,
            "num_layers": 2,
            "forecast_30d": future_denorm,
            "train_size": train_size,
            "test_size": len(test_vals)
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_lstm_forecast.json"
        with open(path, "w") as f:
            json.dump(lstm_results, f, indent=2)

        state["lstm_forecast"] = lstm_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] LSTMForecasterAgent: " +
            "MAE=" + str(round(mae, 4)) +
            " RMSE=" + str(round(rmse, 4))
        )

        self.log("=" * 50)
        self.log("LSTM FORECASTER COMPLETE")
        self.log("MAE  : " + str(round(mae, 4)))
        self.log("RMSE : " + str(round(rmse, 4)))
        self.log("MAPE : " + str(round(mape, 4)) + "%")
        self.log("=" * 50)

        return state