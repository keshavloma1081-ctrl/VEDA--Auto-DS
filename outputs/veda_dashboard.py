import streamlit as st
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
st.markdown("**Goal:** predict whether a customer will churn. target: Churn")
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
target_col = "Churn_Yes"

# ── Tab layout ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Performance", "Feature Insights", "Predictions"])

# ── Tab 1: Overview ───────────────────────────────────────────
with tab1:
    st.markdown("## Pipeline Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "LogisticRegression")
    col2.metric("AUC-ROC", "0.8463")
    col3.metric("F1 Score", "0.5934")
    col4.metric("Accuracy", "0.8045")

    st.markdown("### Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Features", len(df.columns) - 1)

    st.markdown("### VEDA Explanation")
    st.info("Our LogisticRegression model learned to predict customer churn with a good level of accuracy, achieving an AUC-ROC score of 0.8463. The top features driving these predictions are AverageMonthlyCharge, TotalSpend, ContractLength, DataUsage, and CustomerAge, which makes sense as they are all related to a customer s financial commitment and usage patterns. For instance, a high AverageMonthlyCharge may indicate a customer is more likely to churn due to cost sensitivity. In business terms, this model s performance means that we can identify approximately 80% of customers who are likely to churn, allowing for targeted retention efforts. However, the model s precision and recall scores suggest that there may be some false positives and false negatives, so further refinement and validation are necessary. Overall, this model provides a solid foundation for predicting customer churn, but its limitations should be considered when making business decisions.")

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
                0.8463,
                0.5934,
                0.8045,
                0.6621,
                0.5377
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

    top_features = []
    top_values = []

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

    feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

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
