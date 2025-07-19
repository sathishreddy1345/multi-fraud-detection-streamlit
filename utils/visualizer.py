# visualizer.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import io

def plot_bar(model_scores):
    df = pd.DataFrame.from_dict(model_scores, orient="index", columns=["Score"]).sort_values(by="Score", ascending=False)
    st.bar_chart(df)

def plot_shap_summary(model, X_processed):
    st.subheader("🧠 SHAP Explanation")
    try:
        explainer = shap.Explainer(model, X_processed)
        shap_values = explainer(X_processed)

        if X_processed.shape[0] < 2:
            st.warning("⚠️ Not enough rows — using waterfall plot.")
            try:
                shap.plots.waterfall(shap_values[0])
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"❌ Waterfall plot failed: {e}")
        else:
            shap.plots.beeswarm(shap_values)
            st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.error(f"❌ SHAP failed: {e}")

def plot_pie_chart(prob):
    fraud = prob
    not_fraud = 1 - prob
    fig, ax = plt.subplots()
    ax.pie([fraud, not_fraud], labels=["Fraud", "Not Fraud"], autopct="%1.1f%%", colors=["#ff4b4b", "#4caf50"])
    ax.axis("equal")
    st.pyplot(fig)

def plot_confusion_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def get_model_description(key):
    desc = {
        "rf": "🌲 Random Forest: Ensemble of decision trees.",
        "xgb": "🚀 XGBoost: Fast gradient boosting.",
        "lgbm": "🔆 LightGBM: Fast boosting for big data.",
        "cat": "🐱 CatBoost: Good for categorical features.",
        "lr": "📐 Logistic Regression: Simple linear classifier.",
        "iso": "🚨 Isolation Forest: Anomaly detector."
    }
    return desc.get(key, "Unknown model")
