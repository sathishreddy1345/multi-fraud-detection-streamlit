import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap

# 📊 Bar Chart for Model Scores
def plot_bar(scores: dict):
    st.subheader("📈 Model-wise Fraud Confidence")
    df = pd.DataFrame(list(scores.items()), columns=["Model", "Fraud Probability"])
    df = df.sort_values("Fraud Probability", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Fraud Probability", y="Model", palette="magma", ax=ax)
    st.pyplot(fig)


# 🔥 Heatmap for Feature Correlation (if needed)
def plot_heatmap(df: pd.DataFrame):
    st.subheader("🔍 Feature Correlation")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)


# 🧠 SHAP Summary Plot (Safe Version)

def plot_shap_summary(model, X):
    st.subheader("📊 SHAP Explanation")

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        if X.shape[0] == 1:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — using waterfall plot for row 0.")

           try:
    if hasattr(shap_values, 'values') and shap_values.values.ndim == 3:
        st.info("Multi-class model detected — showing class 0 by default.")
        class_idx = 0
        explanation = shap.Explanation(
            values=shap_values.values[0][class_idx],
            base_values=shap_values.base_values[0][class_idx],
            data=shap_values.data[0],
            feature_names=shap_values.feature_names
        )
        shap.plots.waterfall(explanation, show=False)
    else:
        shap.plots.waterfall(shap_values[0], show=False)

    fig = plt.gcf()   # ✅ FIX: get current figure
    st.pyplot(fig)    # ✅ streamlit expects a matplotlib figure here

except Exception as we:
    st.error("❌ Waterfall plot failed:")
    st.code(str(we))
