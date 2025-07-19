import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import io

# 1. All Model Scores Bar Plot with Line
def plot_bar(model_scores):
    st.subheader("📊 All Model Prediction Scores (Bar + Line)")

    df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['Score'])
    df = df.sort_values(by='Score', ascending=False)

    # Combined Bar and Line plot
    fig, ax = plt.subplots()
    df.plot(kind='bar', legend=False, ax=ax, color='#0088cc')
    ax2 = ax.twinx()
    df.plot(marker='o', color='red', legend=False, ax=ax2)
    ax.set_ylabel("Score")
    ax.set_title("Model-wise Fraud Scores")
    st.pyplot(fig)

    # Model Comparison Table
    st.markdown("### 📋 Model Score Table")
    st.dataframe(df.style.background_gradient(cmap='Blues'))

    # Choose a model to drill down
    selected_model = st.selectbox("🔍 Select Model for Insights", df.index.tolist())
    if selected_model:
        st.markdown(f"### 🔎 **{selected_model.upper()}** Insights")
        st.metric("Fraud Confidence Score", f"{model_scores[selected_model]*100:.5f}%")
        st.markdown(get_model_description(selected_model))


# 2. SHAP Summary and Fallback Waterfall
def plot_shap_summary(model, X_processed):
    st.subheader("🧠 SHAP Explanation")

    try:
        explainer = shap.Explainer(model, X_processed)
        shap_values = explainer(X_processed)

        if len(X_processed) < 2:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — showing waterfall for row 0")
            fig = shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(bbox_inches="tight")
        else:
            fig = shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.error(f"❌ SHAP plotting failed: {e}")


# 3. Pie Chart — Works for even small fraud % like 0.00001
def plot_pie_chart(probability_score):
    st.subheader("🥧 Fraud Likelihood Pie Chart")
    fraud = probability_score
    legit = 1 - fraud

    values = [fraud, legit]
    labels = ['Fraud', 'Legit']
    explode = (0.1, 0)

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct=lambda p: '{:.5f}%'.format(p), startangle=90,
           colors=['#e74c3c', '#2ecc71'], explode=explode, shadow=True)
    ax.axis('equal')
    st.pyplot(fig)


# 4. Confusion Matrix + Classification Report
def plot_confusion_report(y_true, y_pred):
    st.subheader("📋 Model Evaluation Report")

    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    st.dataframe(df_report.style.highlight_max(axis=0))

    st.markdown("#### 🔥 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# 5. Feature Heatmap
def plot_feature_heatmap(df):
    st.subheader("🌡️ Feature Correlation Heatmap")

    if df.shape[1] < 2:
        st.info("Need at least 2 numeric features to compute heatmap.")
        return

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


# 6. Description for Models
def get_model_description(model_key):
    descriptions = {
        "rf": "🌲 **Random Forest**: Strong ensemble using many decision trees.",
        "xgb": "🚀 **XGBoost**: High-performance boosting model, fast & accurate.",
        "lgbm": "🔆 **LightGBM**: Gradient booster that's fast on large data.",
        "cat": "🐱 **CatBoost**: Great for categorical features.",
        "lr": "📐 **Logistic Regression**: Simple but interpretable.",
        "iso": "🚨 **Isolation Forest**: Anomaly detector, good for outlier spotting."
    }
    return descriptions.get(model_key, "ℹ️ No description found.")


# Optional SHAP PNG Exporter
def download_shap_summary_as_png(shap_fig, filename="shap_summary.png"):
    buf = io.BytesIO()
    shap_fig.savefig(buf, format="png")
    st.download_button(
        label="📥 Download SHAP Plot",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )
