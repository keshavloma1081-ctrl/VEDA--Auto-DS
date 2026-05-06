"""
VEDA — Autonomous Data Science System
agents/deep_learning/cnn.py — 1D CNN Agent

Builds and trains a 1D Convolutional Neural Network.
Best for sequential/text features and pattern detection.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent


class CNNNetwork(nn.Module):
    """1D CNN for tabular/sequential data."""

    def __init__(self, input_size: int, num_filters: int = 64, dropout: float = 0.3):
        super(CNNNetwork, self).__init__()

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second conv block
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Third conv block
            nn.Conv1d(num_filters * 2, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x


class CNNAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="CNNAgent",
            domain="deep_learning",
            version="1.0.0"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log("Device: " + str(self.device))

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        y = df[target_col].values
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
        y = y.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    def _train_model(self, model, train_loader, val_loader, epochs=40, lr=0.001):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = 8
        history = {"train_loss": [], "val_loss": [], "val_auc": []}

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device).unsqueeze(1)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    all_preds.extend(outputs.cpu().numpy().flatten())
                    all_labels.extend(y_batch.cpu().numpy().flatten())

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except:
                val_auc = 0.5

            history["train_loss"].append(round(avg_train, 4))
            history["val_loss"].append(round(avg_val, 4))
            history["val_auc"].append(round(val_auc, 4))
            scheduler.step(avg_val)

            if epoch % 10 == 0:
                self.log("Epoch " + str(epoch) + "/" + str(epochs) +
                        " loss=" + str(round(avg_train, 4)) +
                        " val_auc=" + str(round(val_auc, 4)))

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log("Early stopping at epoch " + str(epoch))
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model, history

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features for CNN...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        X, y = self._prepare_data(df, target_col)
        self.log("Shape: " + str(X.shape))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        batch_size = min(256, len(X_train) // 10)
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=batch_size
        )

        input_size = X.shape[1]
        num_filters = 32 if input_size < 50 else 64
        self.log("CNN architecture: Conv1D filters=" + str(num_filters))

        model = CNNNetwork(
            input_size=input_size,
            num_filters=num_filters,
            dropout=0.3
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.log("Total parameters: " + str(total_params))

        self.log("Training CNN...")
        model, history = self._train_model(model, train_loader, val_loader, epochs=40)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_pred_proba = model(X_val_t).cpu().numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

        auc = round(float(roc_auc_score(y_val, y_pred_proba)), 4)
        f1 = round(float(f1_score(y_val, y_pred, zero_division=0)), 4)
        acc = round(float(accuracy_score(y_val, y_pred)), 4)

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        model_path = "outputs/" + run_id + "_cnn_model.pt"
        torch.save(model.state_dict(), model_path)

        cnn_results = {
            "model": "CNN1D",
            "filters": num_filters,
            "total_params": total_params,
            "auc": auc,
            "f1": f1,
            "accuracy": acc,
            "epochs_trained": len(history["train_loss"]),
            "model_path": model_path
        }

        results_path = "outputs/" + run_id + "_cnn_results.json"
        with open(results_path, "w") as f:
            json.dump(cnn_results, f, indent=2)

        state.setdefault("dl_results", {})
        state["dl_results"]["cnn"] = cnn_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] CNNAgent: AUC=" + str(auc) +
            " params=" + str(total_params)
        )

        self.log("=" * 50)
        self.log("CNN TRAINING COMPLETE")
        self.log("Filters    : " + str(num_filters))
        self.log("Parameters : " + str(total_params))
        self.log("AUC        : " + str(auc))
        self.log("F1         : " + str(f1))
        self.log("Accuracy   : " + str(acc))
        self.log("=" * 50)

        return state