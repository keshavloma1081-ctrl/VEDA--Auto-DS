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
st.markdown("**Goal:** predict whether a customer review is positive or negative. target: sentiment")
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
target_col = "sentiment"

# ── Tab layout ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Performance", "Feature Insights", "Predictions"])

# ── Tab 1: Overview ───────────────────────────────────────────
with tab1:
    st.markdown("## Pipeline Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "RandomForest")
    col2.metric("AUC-ROC", "1.0")
    col3.metric("F1 Score", "1.0")
    col4.metric("Accuracy", "1.0")

    st.markdown("### Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Features", len(df.columns) - 1)

    st.markdown("### VEDA Explanation")
    st.info("Our RandomForest model learned to predict customer review sentiment with perfect accuracy, indicating that it effectively identified patterns in the data. The model relies heavily on the `review_text` feature, which makes sense since the actual text of the review is the most direct indicator of its sentiment. The `helpful_votes` and `rating` features also contribute, albeit to a much lesser extent, likely because they provide indirect cues about the reviewer s opinion. In business terms, this model s perfect performance means it can accurately classify customer reviews as positive or negative, allowing companies to gauge customer satisfaction and respond accordingly. However, it s worth noting that this exceptional performance may be due to the specific dataset used, and the model may not generalize as well to new, unseen data. Additionally, the model s reliance on `review_text` may make it vulnerable to biases in the language used in the reviews.")

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
                1.0,
                1.0,
                1.0,
                1.0,
                1.0
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

    top_features = ['review_text', 'helpful_votes', 'rating', 'category_Home', 'category_Food', 'category_Toys', 'category_Books', 'category_Beauty']
    top_values = [0.978628, 0.017887, 0.002036, 0.000185, 0.000174, 0.000172, 0.000161, 0.000155]

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

    feature_cols = ['review_text', 'rating', 'helpful_votes', 'category_Beauty', 'category_Books', 'category_Clothing', 'category_Electronics', 'category_Food', 'category_Garden', 'category_Home', 'category_Sports', 'category_Toys']

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
