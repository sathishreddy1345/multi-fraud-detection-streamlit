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
def plot_shap_summary(model, X_input):
    st.subheader("🧠 SHAP Explanation (RandomForest)")
    try:
        # Ensure the input has enough samples (SHAP fails on 1 row often)
        if X_input.shape[0] < 2:
            st.warning("⚠️ SHAP requires at least 2 rows for beeswarm plot.")
            return

        # Use TreeExplainer for tree-based models
        explainer = shap.Explainer(model)
        shap_values = explainer(X_input)

        # Render beeswarm plot
        fig = plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ SHAP summary plot failed.")
        st.code(str(e))
