# app.py
import streamlit as st
import pandas as pd
import time
from fraud_modules import credit_card, paysim, loan, insurance
from utils.visualizer import (
    plot_bar,
    plot_shap_summary,
    plot_pie_chart,
    plot_confusion_report,
    get_model_description,
)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- UI Setup --------------------
st.set_page_config(page_title="🛡️ Multi-Fraud Detector", layout="wide", page_icon="💳")

st.markdown("""
    <style>
    .block-container {
        padding: 2rem;
        border-radius: 12px;
        backdrop-filter: blur(6px);
        background: rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Sidebar Theme --------------------
mode = st.sidebar.radio("🎨 Theme", ["Dark", "Light"])
bg_color = "#1b2735" if mode == "Dark" else "#f5f5f5"
font_color = "#ffffff" if mode == "Dark" else "#111111"

if st.sidebar.button("🔁 Reset App"):
    st.session_state.clear()
    st.rerun()

# -------------------- Page Mapping --------------------
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

# -------------------- Sidebar Info --------------------
st.sidebar.markdown("### 🤖 Model Info")
model_info = {
    "💳 Credit Card": "All 6 models | High accuracy | Balanced performance",
    "📱 PaySim": "IsolationForest + Logistic Regression",
    "🏦 Loan": "LightGBM + Logistic Regression",
    "🚗 Insurance": "CatBoost + RandomForest"
}
for key, val in model_info.items():
    with st.sidebar.expander(key):
        st.write(val)

# -------------------- Navigation --------------------
tabs = ["🏠 Home"] + list(fraud_modules.keys())
selected_tab = st.sidebar.radio("📁 Choose Fraud Type", tabs)

# -------------------- Home --------------------
if selected_tab == "🏠 Home":
    st.title("🛡️ Multi-Fraud Detection System")
    st.info("Upload CSV → Detect Fraud → Visualize & Explain via SHAP & Charts")

# -------------------- Main Prediction Pages --------------------
elif selected_tab in fraud_modules:
    st.title(f"{selected_tab} Fraud Detection")

    uploaded = st.file_uploader("📥 Upload CSV file for prediction", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        if st.button("🔍 Run Prediction"):
            st.success("🧠 Prediction Started...")
            with st.spinner("Analyzing with all models..."):
                predict_fn = getattr(fraud_modules[selected_tab], function_map[selected_tab])
                score, model_scores, X_processed = predict_fn(df)

                # Display final result
                st.success(f"🧠 Final Fraud Score: {score*100:.2f}%")
                plot_pie_chart(score)
                plot_bar(model_scores)

                st.subheader("🔎 Inspect a Model")
                selected_model = st.selectbox("📌 Select a model", list(model_scores.keys()), key="select_model")
                st.metric("Selected Model Score", f"{model_scores[selected_model]*100:.2f}%")

                model = fraud_modules[selected_tab].models.get(selected_model)
                if model:
                    plot_shap_summary(model, X_processed)

                # Optional: show scoreboard
                st.markdown("### 📊 Model Comparison Chart")
                st.bar_chart(pd.DataFrame.from_dict(model_scores, orient='index', columns=['Score']))

                # Confusion Matrix if ground truth present
                if 'actual' in df.columns:
                    y_true = df['actual']
                    y_pred = [1 if model_scores['rf'] > 0.5 else 0] * len(df)
                    plot_confusion_report(y_true, y_pred)

                # Download button
                result_df = df.copy()
                result_df['Fraud_Score'] = score
                st.download_button("⬇️ Download Results", result_df.to_csv(index=False), "fraud_output.csv", "text/csv")
