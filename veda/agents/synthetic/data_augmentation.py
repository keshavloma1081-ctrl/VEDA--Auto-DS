"""
VEDA — Autonomous Data Science System
agents/synthetic/data_augmentation.py — Data Augmentation Agent

Augments training data using:
- SMOTE for class imbalance
- Gaussian noise injection
- Feature mixing
- Bootstrap sampling
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from veda.core.base_agent import BaseAgent


class DataAugmentationAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="DataAugmentationAgent",
            domain="synthetic",
            version="1.0.0"
        )

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _encode_df(self, df):
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=["object", "bool"]).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        return df_enc

    def _smote_augmentation(self, X: np.ndarray,
                             y: np.ndarray) -> tuple:
        """Apply SMOTE for class balancing."""
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            self.log("SMOTE: " + str(len(X)) + " -> " + str(len(X_res)) + " samples")
            return X_res, y_res, "SMOTE"
        except Exception as e:
            self.log("SMOTE failed: " + str(e), level="WARN")
            return X, y, "None"

    def _gaussian_noise(self, X: np.ndarray,
                         noise_factor: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to features."""
        np.random.seed(42)
        noise = np.random.normal(0, noise_factor * X.std(axis=0), X.shape)
        X_noisy = X + noise
        self.log("Gaussian noise added: factor=" + str(noise_factor))
        return X_noisy

    def _feature_mixing(self, X: np.ndarray,
                         alpha: float = 0.2) -> np.ndarray:
        """Mixup augmentation."""
        np.random.seed(42)
        n = len(X)
        indices = np.random.permutation(n)
        lam = np.random.beta(alpha, alpha, n)
        lam = lam.reshape(-1, 1)
        X_mixed = lam * X + (1 - lam) * X[indices]
        self.log("Feature mixing: alpha=" + str(alpha))
        return X_mixed

    def _bootstrap_sampling(self, X: np.ndarray,
                             y: np.ndarray,
                             n_samples: int = None) -> tuple:
        """Bootstrap sampling."""
        np.random.seed(42)
        n = n_samples or len(X)
        indices = np.random.choice(len(X), size=n, replace=True)
        return X[indices], y[indices]

    def run(self, state: dict) -> dict:
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading features for augmentation...")
        df = self._load_features(state)

        if df is None:
            self.log("No features found", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        df_enc = self._encode_df(df)
        feature_cols = [c for c in df.columns if c != target_col]
        X = df_enc[feature_cols].fillna(0).values.astype(np.float32)
        y_raw = df[target_col].values
        if y_raw.dtype == object:
            y_raw = LabelEncoder().fit_transform(y_raw.astype(str))
        y = y_raw.astype(np.float32)

        self.log("Original: " + str(X.shape))

        aug_results = {}

        # SMOTE
        self.log("Applying SMOTE...")
        X_smote, y_smote, smote_method = self._smote_augmentation(X, y)
        aug_results["smote"] = {
            "method": smote_method,
            "original_samples": int(len(X)),
            "augmented_samples": int(len(X_smote)),
            "class_balance": {
                str(cls): int((y_smote == cls).sum())
                for cls in np.unique(y_smote)
            }
        }

        # Gaussian noise
        self.log("Applying Gaussian noise...")
        X_noisy = self._gaussian_noise(X, noise_factor=0.05)
        aug_results["gaussian_noise"] = {
            "noise_factor": 0.05,
            "samples": int(len(X_noisy)),
            "mean_noise": round(float(np.abs(X_noisy - X).mean()), 6)
        }

        # Feature mixing
        self.log("Applying feature mixing...")
        X_mixed = self._feature_mixing(X, alpha=0.2)
        aug_results["feature_mixing"] = {
            "alpha": 0.2,
            "samples": int(len(X_mixed))
        }

        # Bootstrap
        self.log("Applying bootstrap sampling...")
        X_boot, y_boot = self._bootstrap_sampling(X, y, n_samples=len(X) * 2)
        aug_results["bootstrap"] = {
            "original_samples": int(len(X)),
            "bootstrapped_samples": int(len(X_boot))
        }

        augmentation_summary = {
            "original_shape": list(X.shape),
            "augmentation_methods": aug_results,
            "recommended_method": "SMOTE" if smote_method == "SMOTE" else "Bootstrap",
            "final_train_size": int(len(X_smote))
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_augmentation_results.json"
        with open(path, "w") as f:
            json.dump(augmentation_summary, f, indent=2)

        state["augmentation_results"] = augmentation_summary
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] DataAugmentationAgent: " +
            str(len(X)) + " -> " + str(len(X_smote)) + " samples via SMOTE"
        )

        self.log("=" * 50)
        self.log("DATA AUGMENTATION COMPLETE")
        self.log("Original  : " + str(len(X)))
        self.log("After SMOTE: " + str(len(X_smote)))
        self.log("Bootstrap : " + str(len(X_boot)))
        self.log("=" * 50)

        return state