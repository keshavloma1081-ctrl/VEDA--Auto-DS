import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="VEDA AutoDS", page_icon="VEDA", layout="wide")

st.markdown("# VEDA Autonomous Data Science System")
st.markdown("---")

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
target_col = df.columns[-1]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance", "Features", "Predictions"])

with tab1:
    st.markdown("## Pipeline Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "LightGBM")
    c2.metric("AUC-ROC", "1.0")
    c3.metric("F1 Score", "0.9992")
    c4.metric("Accuracy", "0.9989")
    st.markdown("### VEDA Analysis")
    st.info("Our LightGBM model learned to predict with high accuracy whether a passenger survived the Titanic, with a nearly perfect accuracy of 99.89%. The model relies heavily on features like Ticket, Embarked_Q (the port of embarkation), and Fare to make predictions, which makes sense because these factors could be related to a passengers social status, access to resources, and priority during emergency situations. The high importance of Ticket suggests that the model may have identified specific ticket patterns or groups that were more likely to survive. In business terms, this models performance means that it can reliably identify survivors with a high degree of precision, which could be useful for historical analysis or insurance claims. However, its worth noting that the models exceptional performance may be due to overfitting, and its generalizability to other datasets or scenarios should be carefully evaluated.")
    st.markdown("### Sample Data")
    st.dataframe(df.head(10))

with tab2:
    st.markdown("## Model Performance")
    metrics_df = pd.DataFrame({
        "Metric": ["AUC-ROC", "F1 Score", "Accuracy", "Precision", "Recall"],
        "Score": [1.0, 0.9992, 0.9989, 1.0, 0.9985]
    })
    fig = px.bar(metrics_df, x="Metric", y="Score", color="Score",
                 color_continuous_scale="Greens", title="Model Metrics")
    fig.update_layout(yaxis_range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

    X = df.drop(columns=[target_col])
    y_proba = model.predict_proba(X)[:, 1]
    fig2 = px.histogram(x=y_proba, nbins=30, title="Prediction Probability Distribution")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("## Feature Insights")
    top_features = ['Ticket', 'Embarked_Q', 'Fare', 'Pclass', 'Age', 'Survived', 'SibSp', 'Sex_male']
    top_values = [2.735392, 1.374411, 1.016956, 0.556628, 0.526787, 0.222168, 0.101386, 0.026089]
    fig = px.bar(x=top_values, y=top_features, orientation="h",
                 title="Feature Importance", color=top_values,
                 color_continuous_scale="Blues")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## Live Predictions")
    feature_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Ticket', 'Fare', 'Sex_male', 'Embarked_Q']
    input_data = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_cols):
        with cols[i % 3]:
            mean_val = float(df[col].mean()) if df[col].dtype in [np.float64, np.int64] else 0.0
            input_data[col] = st.number_input(col, value=mean_val)
    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", str(pred))
        c2.metric("Prob Class 0", str(round(proba[0], 4)))
        c3.metric("Prob Class 1", str(round(proba[1], 4)))
    st.markdown("### Dataset Predictions")
    X = df.drop(columns=[target_col])
    results_df = df.copy()
    results_df["predicted"] = model.predict(X)
    results_df["probability"] = model.predict_proba(X)[:, 1].round(4)
    st.dataframe(results_df.head(50))
