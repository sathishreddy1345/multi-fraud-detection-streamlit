import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import io

# 📊 1. Compare All Models
def plot_bar(model_scores):
    st.subheader("📊 All Model Prediction Scores")
    df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['Score'])
    df = df.sort_values(by='Score', ascending=False)
    st.bar_chart(df)

    selected_model = st.selectbox("🔍 Select a Model to Inspect", df.index.tolist())
    if selected_model:
        st.markdown(f"### 🔎 {selected_model.upper()} Model Insights")
        st.metric("Fraud Confidence Score", f"{model_scores[selected_model]*100:.2f}%")
        st.bar_chart(pd.DataFrame([model_scores[selected_model]], index=[selected_model], columns=["Score"]))

        # Optional model info
        st.markdown(get_model_description(selected_model))


# 🧠 2. SHAP Summary (or Waterfall fallback)
def plot_shap_summary(model, X_processed):
    st.subheader("🧠 SHAP Explanation")
    try:
        explainer = shap.Explainer(model, X_processed)
        shap_values = explainer(X_processed)

        if len(X_processed) < 2:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — using waterfall plot for row 0.")
            try:
                fig = shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"❌ Waterfall plot failed:\n\n{e}")
        else:
            try:
                fig = shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"❌ SHAP summary plot failed:\n\n{e}")
    except Exception as e:
        st.error(f"❌ SHAP explanation failed:\n\n{e}")


# 🥧 3. Pie Chart — Estimated Fraud Risk
def plot_pie_chart(probability_score):
    st.subheader("📊 Estimated Fraud Likelihood")

    fraud_pct = probability_score
    labels = ['Fraud', 'Not Fraud']
    values = [fraud_pct, 1 - fraud_pct]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff4b4b', '#4caf50'])
    ax.axis('equal')
    st.pyplot(fig)


# 📋 4. Confusion Matrix and Report (optional, needs true labels)
def plot_confusion_report(y_true, y_pred):
    st.subheader("📊 Model Evaluation Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df.style.highlight_max(axis=0))

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# 🧾 5. Model Descriptions
def get_model_description(model_key):
    descriptions = {
        "rf": "🌲 **Random Forest**: A powerful ensemble model using multiple decision trees. Great for tabular data.",
        "xgb": "🚀 **XGBoost**: Gradient boosting algorithm that optimizes model accuracy and speed.",
        "lgbm": "🔆 **LightGBM**: A fast and efficient boosting framework optimized for large datasets.",
        "cat": "🐱 **CatBoost**: Gradient boosting from Yandex, handles categorical data well.",
        "lr": "📐 **Logistic Regression**: Simple, interpretable linear model for binary classification.",
        "iso": "🚨 **Isolation Forest**: Anomaly detection model—flags outliers as potential frauds."
    }
    return descriptions.get(model_key, "No description available.")


# 🖨️ 6. Download Button for SHAP (Optional)
def download_shap_summary_as_png(shap_fig, filename="shap_summary.png"):
    buf = io.BytesIO()
    shap_fig.savefig(buf, format="png")
    st.download_button(
        label="📥 Download SHAP Plot",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )
