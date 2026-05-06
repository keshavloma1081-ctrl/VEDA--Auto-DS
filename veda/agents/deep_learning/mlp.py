"""
VEDA — Autonomous Data Science System
agents/deep_learning/mlp.py — Multi-Layer Perceptron Agent

Builds and trains a PyTorch MLP on tabular data.
Auto-configures architecture based on input size.
"""

import os
import json
import joblib
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


class MLPNetwork(nn.Module):
    """Auto-configured MLP based on input size."""

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout: float = 0.3):
        super(MLPNetwork, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        if output_size == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="MLPAgent",
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

    def _auto_architecture(self, input_size: int) -> list:
        """Auto-configure hidden layer sizes based on input size."""
        if input_size < 20:
            return [64, 32]
        elif input_size < 50:
            return [128, 64, 32]
        elif input_size < 100:
            return [256, 128, 64]
        else:
            return [512, 256, 128, 64]

    def _prepare_data(self, df, target_col):
        """Prepare data for PyTorch."""
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        y = df[target_col].values

        # Encode target if needed
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)

        y = y.astype(np.float32)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    def _train_model(self, model, train_loader, val_loader,
                     epochs: int = 30, lr: float = 0.001):
        """Training loop with early stopping."""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        patience = 10

        history = {"train_loss": [], "val_loss": [], "val_auc": []}

        for epoch in range(epochs):
            # Training
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

            # Validation
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

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except:
                val_auc = 0.5

            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["val_auc"].append(round(val_auc, 4))

            scheduler.step(avg_val_loss)

            if epoch % 10 == 0:
                self.log("Epoch " + str(epoch) + "/" + str(epochs) +
                        " — train_loss=" + str(round(avg_train_loss, 4)) +
                        " val_loss=" + str(round(avg_val_loss, 4)) +
                        " val_auc=" + str(round(val_auc, 4)))

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log("Early stopping at epoch " + str(epoch))
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)

        return model, history

    def run(self, state: dict) -> dict:
        """
        MLP Training:
        1. Load features
        2. Auto-configure architecture
        3. Train with early stopping
        4. Evaluate
        5. Save model
        """

        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]
            self.log("Using last column as target: " + str(target_col), level="WARN")

        self.log("Preparing data...")
        X, y = self._prepare_data(df, target_col)
        self.log("Shape: " + str(X.shape) + " target classes: " + str(np.unique(y).tolist()))

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create DataLoaders
        batch_size = min(256, len(X_train) // 10)
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Auto-configure architecture
        input_size = X.shape[1]
        hidden_sizes = self._auto_architecture(input_size)
        self.log("Architecture: " + str(input_size) + " -> " + str(hidden_sizes) + " -> 1")

        # Build model
        model = MLPNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            dropout=0.3
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.log("Total parameters: " + str(total_params))

        # Train
        self.log("Training MLP...")
        model, history = self._train_model(
            model, train_loader, val_loader,
            epochs=50, lr=0.001
        )

        # Final evaluation
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_pred_proba = model(X_val_tensor).cpu().numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

        auc = round(float(roc_auc_score(y_val, y_pred_proba)), 4)
        f1 = round(float(f1_score(y_val, y_pred)), 4)
        acc = round(float(accuracy_score(y_val, y_pred)), 4)
        best_val_auc = max(history["val_auc"]) if history["val_auc"] else 0

        self.log("Final AUC : " + str(auc))
        self.log("Final F1  : " + str(f1))
        self.log("Final Acc : " + str(acc))

        # Save model
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        model_path = "outputs/" + run_id + "_mlp_model.pt"
        torch.save(model.state_dict(), model_path)
        self.log("Model saved to: " + model_path)

        # Save architecture config
        config = {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "output_size": 1,
            "dropout": 0.3,
            "total_params": total_params,
            "device": str(self.device)
        }
        config_path = "outputs/" + run_id + "_mlp_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save results
        dl_results = {
            "model": "MLP",
            "architecture": str(input_size) + " -> " + str(hidden_sizes) + " -> 1",
            "total_params": total_params,
            "auc": auc,
            "f1": f1,
            "accuracy": acc,
            "best_val_auc": best_val_auc,
            "epochs_trained": len(history["train_loss"]),
            "model_path": model_path,
            "history": history
        }

        results_path = "outputs/" + run_id + "_mlp_results.json"
        with open(results_path, "w") as f:
            json.dump(dl_results, f, indent=2)

        state["dl_results"] = dl_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] MLPAgent: AUC=" + str(auc) +
            " F1=" + str(f1) + " params=" + str(total_params)
        )

        self.log("=" * 50)
        self.log("MLP TRAINING COMPLETE")
        self.log("Architecture : " + str(input_size) + " -> " + str(hidden_sizes) + " -> 1")
        self.log("Parameters   : " + str(total_params))
        self.log("AUC          : " + str(auc))
        self.log("F1           : " + str(f1))
        self.log("Accuracy     : " + str(acc))
        self.log("=" * 50)

        return state