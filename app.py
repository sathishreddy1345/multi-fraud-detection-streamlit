# app.py — AI-Powered Fraud Detection System with Advanced Visualization

import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from fraud_modules import credit_card, paysim, loan, insurance

from utils.visualizer import (
    plot_bar, plot_shap_summary, plot_pie_chart, plot_confusion_report,get_model_description,
    plot_boxplot, plot_radar, download_model_report, plot_correlation_heatmap,plot_shap_force
)


from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="🛡️ Multi-Fraud Detection System",
    layout="wide",
    page_icon="🧠"
)

# 🎨 Custom Styling & Animations
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #1f1c2c, #928DAB, #0f2027);
    background-size: 400% 400%;
    animation: gradient 20s ease infinite;
    color: #f5f5f5;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.block-container {
    backdrop-filter: blur(8px);
    background-color: rgba(0, 0, 0, 0.25);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧭 Navigation Panel")
tabs = ["🏠 Home", "💳 Credit Card", "📱 PaySim", "🏦 Loan", "🚗 Insurance"]
selected_tab = st.sidebar.radio("Select Fraud Type", tabs)

if "Model Info" not in st.session_state:
    st.session_state["Model Info"] = {
        "💳 Credit Card": "RandomForest, XGBoost, CatBoost, LightGBM, Logistic Regression, IsolationForest",
        "📱 PaySim": "Logistic Regression + IsolationForest",
        "🏦 Loan": "LightGBM + Logistic Regression",
        "🚗 Insurance": "CatBoost + Random Forest"
    }

with st.sidebar.expander("📘 Model Details", expanded=False):
    for model, desc in st.session_state["Model Info"].items():
        st.markdown(f"**{model}**: {desc}")

if st.sidebar.button("🔁 Reset App"):
    st.session_state.clear()
    st.experimental_rerun()

# Fraud prediction modules
fraud_modules = {
    "💳 Credit Card": credit_card,
    "📱 PaySim": paysim,
    "🏦 Loan": loan,
    "🚗 Insurance": insurance
}
function_map = {
    "💳 Credit Card": "predict_creditcard_fraud",
    "📱 PaySim": "predict_paysim_fraud",
    "🏦 Loan": "predict_loan_fraud",
    "🚗 Insurance": "predict_insurance_fraud"
}

# 🏠 Home
if selected_tab == "🏠 Home":
    st.title("🛡️ Multi-Fraud Detection Dashboard")
    st.markdown("Welcome! Choose a fraud type from the sidebar or buttons below.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💳 Credit Card Fraud"):
            st.session_state["page"] = "💳 Credit Card"
            st.experimental_rerun()
        if st.button("🏦 Loan Fraud"):
            st.session_state["page"] = "🏦 Loan"
            st.experimental_rerun()
    with col2:
        if st.button("📱 PaySim Fraud"):
            st.session_state["page"] = "📱 PaySim"
            st.experimental_rerun()
        if st.button("🚗 Insurance Fraud"):
            st.session_state["page"] = "🚗 Insurance"
            st.experimental_rerun()

# 🚨 Main Fraud Pages
if selected_tab in fraud_modules:
    st.title(f"{selected_tab} Detection")
    uploaded = st.file_uploader("📤 Upload a CSV file for analysis", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(5), height=250)

        if st.button("🔍 Run AI Fraud Detection"):
            with st.spinner("Analyzing... please wait ⏳"):
                time.sleep(1)
                fn = function_map[selected_tab]
                score, model_scores, processed = getattr(fraud_modules[selected_tab], fn)(df)

            # 📊 Bar chart of all model scores
            selected_model = plot_bar(model_scores)

            # 🧠 SHAP Explanation (default to RF if present)
            default_model = fraud_modules[selected_tab].models.get("rf") or list(fraud_modules[selected_tab].models.values())[0]
            plot_shap_summary(default_model, processed)

            # 🥧 Pie Chart
            plot_pie_chart(score)

            st.success(f"✅ Overall Fraud Likelihood: **{score*100:.2f}%**")

            # 🔎 Inspect individual model
            st.markdown("### 🔬 Explore Individual Model")
            selected_model = st.selectbox("Choose a model", list(model_scores.keys()))
            st.metric("Score", f"{model_scores[selected_model]*100:.2f}%")
            st.markdown(get_model_description(selected_model))

            # 📥 Export
            st.download_button("⬇️ Download Prediction CSV", df.to_csv(index=False), file_name="results.csv")

            # Optional Confusion Matrix
            if 'actual' in df.columns:
                y_true = df['actual']
                y_pred = [1 if model_scores[selected_model] > 0.5 else 0]*len(df)
                plot_confusion_report(y_true, y_pred)


            plot_radar(model_scores)
          
            plot_boxplot(processed)
            
            try:
                selected_model_object = models[selected_model]
                plot_shap_force(selected_model_object, processed)
            except Exception as e:
                st.warning(f"⚠️ SHAP force plot not available: {e}")

            plot_correlation_heatmap(processed)
            
            download_model_report(processed)

