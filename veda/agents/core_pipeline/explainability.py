import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from veda.core.base_agent import BaseAgent

load_dotenv()

class ExplainabilityAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ExplainabilityAgent", domain="ml", version="1.0.0")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _load_model(self, state):
        model_info = state.get("model_info", {})
        model_path = model_info.get("model_path")
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_model.pkl")]
        return joblib.load(os.path.join(d, sorted(files)[-1]))

    def _get_feature_importance(self, model, feature_names):
        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                pairs = list(zip(feature_names, importance))
                pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
                return {name: round(float(val), 6) for name, val in pairs_sorted}
        except Exception as e:
            self.log("Feature importance failed: " + str(e), level="WARN")
        return {}

    def _try_shap(self, model, X):
        try:
            import shap
            self.log("Computing SHAP values...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.head(100))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(X.columns, [round(float(v), 6) for v in mean_shap]))
            shap_sorted = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
            return shap_sorted
        except ImportError:
            self.log("SHAP not installed — using feature importance instead", level="WARN")
            return None
        except Exception as e:
            self.log("SHAP failed: " + str(e) + " — using feature importance", level="WARN")
            return None

    def _generate_explanation(self, goal, feature_importance, metrics, model_name):
        top_features = list(feature_importance.keys())[:5]
        top_values = [feature_importance[f] for f in top_features]

        prompt = """You are VEDA, an autonomous data science system.

You just trained a """ + model_name + """ model for this goal: """ + goal + """

Model metrics:
""" + json.dumps(metrics, indent=2) + """

Top 5 most important features:
""" + "\n".join([str(i+1) + ". " + top_features[i] + " (importance=" + str(top_values[i]) + ")" for i in range(len(top_features))]) + """

Write a clear 4-6 sentence plain-English explanation covering:
1. What the model learned overall
2. Which features drive predictions most and why that makes sense
3. What the model performance means in business terms
4. Any caveats or limitations

Be specific and mention actual feature names."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are VEDA, an expert data scientist. Be concise and clear."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()

    def run(self, state):
        goal = state.get("goal", "")
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        model_info = state.get("model_info", {})
        model_name = model_info.get("model_name", "LightGBM")
        metrics = model_info.get("test_metrics", {})

        self.log("Loading model and features...")
        df = self._load_features(state)
        model = self._load_model(state)

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        X = df.drop(columns=[target_col])
        feature_names = list(X.columns)

        # Try SHAP first, fall back to feature importance
        self.log("Computing feature importance...")
        shap_importance = self._try_shap(model, X)

        if shap_importance:
            feature_importance = shap_importance
            method = "SHAP"
        else:
            feature_importance = self._get_feature_importance(model, feature_names)
            method = "Feature Importance"

        self.log("Method: " + method)
        top5 = list(feature_importance.items())[:5]
        for name, val in top5:
            self.log("  " + name + ": " + str(val))

        # Generate plain-English explanation
        self.log("Generating explanation with Groq...")
        explanation = self._generate_explanation(goal, feature_importance, metrics, model_name)

        # Save results
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        explainability_path = "outputs/" + run_id + "_explainability.json"
        explainability_data = {
            "method": method,
            "feature_importance": feature_importance,
            "top_features": list(feature_importance.keys())[:10],
            "explanation_text": explanation
        }
        with open(explainability_path, "w") as f:
            json.dump(explainability_data, f, indent=2)

        # Update state
        state["explainability"] = {
            "top_features": list(feature_importance.keys())[:5],
            "explanation_text": explanation,
            "feature_importance": feature_importance
        }

        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ExplainabilityAgent: top feature=" +
            list(feature_importance.keys())[0]
        )

        self.log("=" * 50)
        self.log("EXPLAINABILITY COMPLETE")
        self.log("Method: " + method)
        self.log("Top features: " + str(list(feature_importance.keys())[:5]))
        self.log("Explanation:")
        self.log(explanation)
        self.log("=" * 50)

        return state