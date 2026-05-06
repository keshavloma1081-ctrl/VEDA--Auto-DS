"""
VEDA — Autonomous Data Science System
agents/nlp/text_classification.py — Text Classification Agent

Classifies text using:
- Logistic Regression on TF-IDF (fast baseline)
- LightGBM on TF-IDF (strong performer)
- Zero-shot classification via Groq LLM
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

from veda.core.base_agent import BaseAgent

load_dotenv()


class TextClassificationAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="TextClassificationAgent",
            domain="nlp",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_tfidf_features(self, state):
        """Load TF-IDF feature matrix."""
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_tfidf_features.parquet")]
        if not files:
            self.log("No TF-IDF features found — using raw features", level="WARN")
            files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
            if not files:
                return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _train_logistic(self, X, y):
        """Train Logistic Regression on TF-IDF features."""
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver="lbfgs",
            multi_class="auto"
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        model.fit(X, y)
        return model, scores.mean(), scores.std()

    def _train_lightgbm(self, X, y):
        """Train LightGBM on TF-IDF features."""
        model = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        model.fit(X, y)
        return model, scores.mean(), scores.std()

    def _zero_shot_classify(self, texts: list, labels: list) -> list:
        """Use Groq LLM for zero-shot classification on small samples."""
        results = []
        for text in texts[:5]:  # Only classify 5 samples as demo
            prompt = """Classify this text into one of these categories: """ + str(labels) + """

Text: """ + str(text)[:500] + """

Reply with ONLY the category name, nothing else."""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=20
                )
                results.append(response.choices[0].message.content.strip())
            except Exception as e:
                results.append("unknown")
        return results

    def run(self, state: dict) -> dict:
        """
        Text classification:
        1. Load TF-IDF features
        2. Train Logistic Regression
        3. Train LightGBM
        4. Compare and select winner
        5. Zero-shot demo with Groq
        """

        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading TF-IDF features...")
        df = self._load_tfidf_features(state)

        if df is None:
            self.log("No features available for text classification", level="WARN")
            return state

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]
            self.log("Using last column as target: " + str(target_col), level="WARN")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.log("Feature matrix: " + str(X.shape))
        self.log("Target: " + str(target_col))
        self.log("Classes: " + str(y.unique().tolist()))

        results = {}

        # Train Logistic Regression
        self.log("Training Logistic Regression on TF-IDF...")
        try:
            lr_model, lr_auc, lr_std = self._train_logistic(X, y)
            results["LogisticRegression"] = {"auc": round(float(lr_auc), 4), "std": round(float(lr_std), 4)}
            self.log("LogisticRegression AUC: " + str(round(lr_auc, 4)) + " +/- " + str(round(lr_std, 4)))
        except Exception as e:
            self.log("LogisticRegression failed: " + str(e), level="WARN")
            lr_model = None

        # Train LightGBM
        self.log("Training LightGBM on TF-IDF...")
        try:
            lgb_model, lgb_auc, lgb_std = self._train_lightgbm(X, y)
            results["LightGBM"] = {"auc": round(float(lgb_auc), 4), "std": round(float(lgb_std), 4)}
            self.log("LightGBM AUC: " + str(round(lgb_auc, 4)) + " +/- " + str(round(lgb_std, 4)))
        except Exception as e:
            self.log("LightGBM failed: " + str(e), level="WARN")
            lgb_model = None

        # Select winner
        best_model_name = max(results, key=lambda k: results[k]["auc"]) if results else "LogisticRegression"
        best_model = lr_model if best_model_name == "LogisticRegression" else lgb_model
        best_auc = results.get(best_model_name, {}).get("auc", 0)

        self.log("Winner: " + best_model_name + " AUC=" + str(best_auc))

        # Save best model
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        model_path = "outputs/" + run_id + "_nlp_model.pkl"
        if best_model:
            joblib.dump(best_model, model_path)
            self.log("NLP model saved to: " + model_path)

        # Zero-shot demo
        self.log("Running zero-shot classification demo with Groq...")
        try:
            cleaned_files = [f for f in os.listdir("outputs") if f.endswith("_cleaned_text.parquet")]
            if cleaned_files:
                cleaned_df = pd.read_parquet(os.path.join("outputs", sorted(cleaned_files)[-1]))
                sample_texts = cleaned_df["cleaned_text"].head(5).tolist()
                unique_labels = y.unique().tolist()
                zero_shot_results = self._zero_shot_classify(sample_texts, unique_labels)
                self.log("Zero-shot results (5 samples): " + str(zero_shot_results))
        except Exception as e:
            self.log("Zero-shot demo failed: " + str(e), level="WARN")
            zero_shot_results = []

        # Save results
        classification_results = {
            "benchmark": results,
            "winner": best_model_name,
            "best_auc": best_auc,
            "model_path": model_path,
            "zero_shot_demo": zero_shot_results
        }

        results_path = "outputs/" + run_id + "_text_classification.json"
        with open(results_path, "w") as f:
            json.dump(classification_results, f, indent=2)

        state["text_classification"] = classification_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] TextClassificationAgent: " +
            best_model_name + " AUC=" + str(best_auc)
        )

        self.log("=" * 50)
        self.log("TEXT CLASSIFICATION COMPLETE")
        self.log("Winner  : " + best_model_name)
        self.log("AUC     : " + str(best_auc))
        self.log("=" * 50)

        return state