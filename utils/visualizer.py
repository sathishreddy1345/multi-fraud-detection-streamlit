# utils/visualizer.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_bar(model_scores):
    st.subheader("📊 AI Model Scores")
    df = pd.DataFrame(model_scores.items(), columns=['Model', 'Score']).set_index("Model")
    st.bar_chart(df)

def plot_shap_summary(model, X):
    st.subheader("🧠 SHAP Explanation")
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        if len(X) < 2:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — using waterfall plot.")
            fig = shap.plots.waterfall(shap_values[0], show=False)
        else:
            fig = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.error(f"❌ SHAP failed: {e}")

def plot_pie_chart(probability_score):
    st.subheader("🥧 Fraud Probability Pie Chart")
    labels = ['Fraud', 'Not Fraud']
    values = [probability_score, 1 - probability_score]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#e63946', '#2a9d8f'], startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

def get_model_description(model_key):
    return {
        "rf": "🌲 Random Forest: Ensemble of decision trees.",
        "xgb": "🚀 XGBoost: Gradient boosting that’s fast and accurate.",
        "lgbm": "🔆 LightGBM: Light and fast gradient boosting.",
        "cat": "🐱 CatBoost: Best for categorical features.",
        "lr": "📐 Logistic Regression: Classic baseline classifier.",
        "iso": "🚨 Isolation Forest: Detects outliers and anomalies."
    }.get(model_key, "No description available.")

def plot_confusion_report(y_true, y_pred):
    st.subheader("📊 Evaluation Metrics")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
