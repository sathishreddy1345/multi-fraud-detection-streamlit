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


import streamlit as st
import shap
import matplotlib.pyplot as plt

def plot_shap_summary(model, X):
    st.subheader("📊 SHAP Explanation")

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        if X.shape[0] == 1:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — using waterfall plot for row 0.")

            # 🧠 Handle multi-class or multi-output predictions
            if len(shap_values.shape) == 3:
                # (samples, classes, features)
                st.info("Multi-class model detected — showing class 0 by default")
                fig = shap.plots.waterfall(shap_values[0, 0], show=False)
            else:
                fig = shap.plots.waterfall(shap_values[0], show=False)

            st.pyplot(fig)

        else:
            fig = shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)

    except Exception as e:
        st.error("❌ SHAP plot failed:")
        st.code(str(e))
