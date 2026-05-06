import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from veda.core.base_agent import BaseAgent

class DashboardAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="DashboardAgent", domain="dashboard", version="1.0.0")

    def _load_features(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_cleaned.parquet")]
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _load_model(self, state):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_model.pkl")]
        if not files:
            raise FileNotFoundError("No model found")
        return joblib.load(os.path.join(d, sorted(files)[-1]))

    def _generate_dashboard(self, state, df, model, target_col, metrics, feature_importance, explanation, run_id):
        """Generate a Streamlit dashboard Python file."""

        goal = state.get("goal", "")
        model_name = state.get("model_info", {}).get("model_name", "LightGBM")
        top_features = list(feature_importance.keys())[:8]
        top_values = [feature_importance[f] for f in top_features]
        feature_cols = [c for c in df.columns if c != target_col]

        dashboard_code = '''import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(
    page_title="VEDA — AutoDS Results",
    page_icon="🧠",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🧠 VEDA — Autonomous Data Science System")
st.markdown("**Goal:** ''' + goal + '''")
st.markdown("---")

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    d = "outputs"
    files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
    return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

@st.cache_resource
def load_model():
    d = "outputs"
    files = [f for f in os.listdir(d) if f.endswith("_model.pkl")]
    return joblib.load(os.path.join(d, sorted(files)[-1]))

df = load_data()
model = load_model()
target_col = "''' + str(target_col) + '''"

# ── Tab layout ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Performance", "Feature Insights", "Predictions"])

# ── Tab 1: Overview ───────────────────────────────────────────
with tab1:
    st.markdown("## Pipeline Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "''' + model_name + '''")
    col2.metric("AUC-ROC", "''' + str(metrics.get("auc_roc", "N/A")) + '''")
    col3.metric("F1 Score", "''' + str(metrics.get("f1_score", "N/A")) + '''")
    col4.metric("Accuracy", "''' + str(metrics.get("accuracy", "N/A")) + '''")

    st.markdown("### Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Features", len(df.columns) - 1)

    st.markdown("### VEDA Explanation")
    st.info("''' + explanation.replace("'", " ").replace("\n", " ") + '''")

    st.markdown("### Sample Data")
    st.dataframe(df.head(10))

# ── Tab 2: Model Performance ──────────────────────────────────
with tab2:
    st.markdown("## Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["AUC-ROC", "F1 Score", "Accuracy", "Precision", "Recall"],
            "Score": [
                ''' + str(metrics.get("auc_roc", 0)) + ''',
                ''' + str(metrics.get("f1_score", 0)) + ''',
                ''' + str(metrics.get("accuracy", 0)) + ''',
                ''' + str(metrics.get("precision", 0)) + ''',
                ''' + str(metrics.get("recall", 0)) + '''
            ]
        })
        fig = px.bar(metrics_df, x="Metric", y="Score",
                     color="Score", color_continuous_scale="Greens",
                     title="Model Metrics")
        fig.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Prediction Distribution")
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            y_proba = model.predict_proba(X)[:, 1]
            fig2 = px.histogram(x=y_proba, nbins=30,
                                title="Prediction Probability Distribution",
                                labels={"x": "Predicted Probability"})
            st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Feature Insights ───────────────────────────────────
with tab3:
    st.markdown("## Feature Insights")

    top_features = ''' + str(top_features) + '''
    top_values = ''' + str(top_values) + '''

    fig = px.bar(
        x=top_values, y=top_features,
        orientation="h",
        title="Top Feature Importance",
        labels={"x": "Importance", "y": "Feature"},
        color=top_values,
        color_continuous_scale="Blues"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        fig3 = px.imshow(corr, title="Feature Correlation Matrix",
                         color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: Predictions ────────────────────────────────────────
with tab4:
    st.markdown("## Live Predictions")
    st.markdown("Enter feature values below to get a prediction:")

    feature_cols = ''' + str(feature_cols) + '''

    input_data = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_cols):
        with cols[i % 3]:
            if df[col].dtype in [np.float64, np.int64]:
                val = st.number_input(col, value=float(df[col].mean()))
            else:
                val = st.text_input(col, value=str(df[col].mode()[0]))
            input_data[col] = val

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([input_data])
        for col in feature_cols:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                pass
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", str(pred))
        col2.metric("Probability Class 0", str(round(proba[0], 4)))
        col3.metric("Probability Class 1", str(round(proba[1], 4)))

    st.markdown("### All Predictions on Dataset")
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        preds = model.predict(X)
        probas = model.predict_proba(X)[:, 1]
        results_df = df.copy()
        results_df["predicted"] = preds
        results_df["probability"] = probas.round(4)
        st.dataframe(results_df.head(50))
'''

        return dashboard_code

    def run(self, state):
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None
        model_info = state.get("model_info", {})
        metrics = model_info.get("test_metrics", {})
        explainability = state.get("explainability", {})
        feature_importance = explainability.get("feature_importance", {})
        explanation = explainability.get("explanation_text", "No explanation available.")

        self.log("Loading data and model...")
        df = self._load_features(state)
        model = self._load_model(state)

        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        self.log("Generating Streamlit dashboard...")
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        dashboard_code = self._generate_dashboard(
            state, df, model, target_col,
            metrics, feature_importance, explanation, run_id
        )

        os.makedirs("outputs", exist_ok=True)
        dashboard_path = "outputs/veda_dashboard.py"
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(dashboard_code)

        state.setdefault("outputs", {})
        state["outputs"]["dashboard_path"] = dashboard_path

        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] DashboardAgent: dashboard saved to " + dashboard_path
        )

        self.log("Dashboard saved to: " + dashboard_path)
        self.log("Run with: streamlit run " + dashboard_path)

        return state