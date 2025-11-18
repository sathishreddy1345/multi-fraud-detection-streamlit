import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

# Fraud modules
from fraud_modules import credit_card, paysim, loan, insurance

# Visualizers
from utils.visualizer import (
    plot_bar, plot_feature_importance, plot_permutation_importance,
    plot_pie_chart, plot_confusion_report, get_model_description,
    plot_boxplot, plot_radar, download_model_report, plot_correlation_heatmap
)

import json

def load_metrics():
    try:
        with open("models/credit_card_metrics.json") as f:
            return json.load(f)
    except Exception as e:
        print("❌ Metrics load error:", e)
        return {}

# Page setup
st.set_page_config(page_title="🛡️ Multi-Fraud Detection System", layout="wide", page_icon="🧠")

# Styling
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #1f1c2c, #928DAB, #0f2027);
    background-size: 400% 400%;
    animation: gradient 20s ease infinite;
    color: #f5f5f5;
}
.block-container {
    backdrop-filter: blur(8px);
    background-color: rgba(0, 0, 0, 0.25);
    border-radius: 20px;
    padding: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Tabs
tabs = ["🏠 Home", "💳 Credit Card", "📱 PaySim", "🏦 Loan", "🚗 Insurance"]

if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = "🏠 Home"

selected_tab = st.sidebar.radio("Select Fraud Type", tabs, index=tabs.index(st.session_state["selected_tab"]))

# Sidebar
st.sidebar.title("🧭 Navigation Panel")
if "Model Info" not in st.session_state:
    st.session_state["Model Info"] = {
        "💳 Credit Card": "RandomForest, XGBoost, CatBoost, Logistic Regression, IsolationForest",
        "📱 PaySim": "Logistic Regression + IsolationForest",
        "🏦 Loan": "XGBOOST + Logistic Regression",
        "🚗 Insurance": "CatBoost + Random Forest"
    }

with st.sidebar.expander("📘 Model Details"):
    for model, desc in st.session_state["Model Info"].items():
        st.markdown(f"**{model}**: {desc}")

if st.sidebar.button("🔁 Reset App"):
    st.session_state.clear()
    st.experimental_rerun()

# Module maps
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

# HOME PAGE
if selected_tab == "🏠 Home":
    st.title("🛡️ Multi-Fraud Detection Dashboard")
    st.markdown("Choose a fraud type from the sidebar or buttons below.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💳 Credit Card Fraud"): st.session_state["selected_tab"] = "💳 Credit Card"; st.experimental_rerun()
        if st.button("🏦 Loan Fraud"): st.session_state["selected_tab"] = "🏦 Loan"; st.experimental_rerun()
    with col2:
        if st.button("📱 PaySim Fraud"): st.session_state["selected_tab"] = "📱 PaySim"; st.experimental_rerun()
        if st.button("🚗 Insurance Fraud"): st.session_state["selected_tab"] = "🚗 Insurance"; st.experimental_rerun()


# =============================
#   PREDICTION PAGES
# =============================
if selected_tab in fraud_modules:
    st.title(f"{selected_tab} Detection")
    uploaded = st.file_uploader("📤 Upload CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded, thousands=",")
        for col in df.columns:
            if df[col].dtype == object:
                cleaned = df[col].astype(str).str.replace(",", "").str.replace("₹", "")
                try: df[col] = pd.to_numeric(cleaned)
                except: df[col] = cleaned

        df = df.dropna(axis=1, how="all")
        edited_df = st.data_editor(df.head(50), use_container_width=True, num_rows="dynamic", key="editor_input")

        if st.button("🔍 Run AI Fraud Detection"):
            df = edited_df
            with st.spinner("Analyzing..."):
                try:
                    fn = function_map[selected_tab]
                    score, model_scores, processed = getattr(fraud_modules[selected_tab], fn)(df)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                st.session_state["model_scores"] = model_scores
                st.session_state["score"] = score
                st.session_state["processed"] = processed
                st.session_state["uploaded_df"] = df

    # Show results
    if "model_scores" in st.session_state and st.session_state["selected_tab"] == selected_tab:

        model_scores = st.session_state["model_scores"]
        score = st.session_state["score"]
        processed = st.session_state["processed"]
        df = st.session_state["uploaded_df"]

        st.markdown("## 🔍 Research Ensemble Score (Soft Voting)")
        model_list = list(model_scores.keys())
        scores_array = np.array(list(model_scores.values()))
        weights = np.ones(len(model_list))
        ensemble_research = np.sum(weights * scores_array) / np.sum(weights)
        st.metric("📌 Ensemble Fraud Likelihood", f"{ensemble_research*100:.2f}%")

        # Show ensemble table
        df_w = pd.DataFrame({"Model": model_list, "Prediction Score": scores_array, "Weight": weights})
        st.dataframe(df_w.style.format({"Prediction Score": "{:.4f}", "Weight": "{:.2f}"}))

        # ===========================
        #     METRICS SECTION
        # ===========================
        metrics = load_metrics()

        if metrics:
            st.markdown("## 🏆 Best Model (Based on F1-score)")
            best_model_key = max(metrics.keys(), key=lambda m: metrics[m]["f1"])

            name_map = {
                "rf": "Random Forest",
                "xgb": "XGBoost",
                "cat": "CatBoost",
                "lr": "Logistic Regression",
                "iso": "IsolationForest"
            }
            st.success(f"⭐ Best Model: **{name_map[best_model_key]}** — F1 Score: **{metrics[best_model_key]['f1']:.4f}**")

            st.markdown("---")
            st.markdown("## 📊 Credit Card Fraud Model Metrics")

            selected_model_name = st.selectbox("🔽 Select Model", list(name_map.values()))
            key = {v:k for k,v in name_map.items()}[selected_model_name]

            model_metrics = metrics[key]

            st.markdown("### 🧮 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{model_metrics['precision']:.4f}")
            col3.metric("Recall", f"{model_metrics['recall']:.4f}")

            col4, col5 = st.columns(2)
            col4.metric("F1 Score", f"{model_metrics['f1']:.4f}")
            col5.metric("ROC-AUC", f"{model_metrics['roc_auc']:.4f}" if model_metrics["roc_auc"] else "N/A")

            st.markdown("### 🔲 Confusion Matrix")
            cm_df = pd.DataFrame(model_metrics["confusion_matrix"], 
                                 columns=["Pred 0", "Pred 1"], 
                                 index=["Actual 0", "Actual 1"])
            st.dataframe(cm_df)

            st.markdown("### 📄 Classification Report")
            st.text(model_metrics["classification_report"])

        # ===========================
        #   MODEL INSPECTION
        # ===========================
        all_models = fraud_modules[selected_tab].models
        all_models_full = fraud_modules[selected_tab].models_full

        st.success(f"🎯 Overall Fraud Likelihood: {score*100:.2f}%")

        st.markdown("### 🔬 Explore Individual Model")
        selected_model = st.selectbox("Choose Model", list(model_scores.keys()))
        st.metric("Model Score", f"{model_scores[selected_model]*100:.2f}%")
        st.markdown(get_model_description(selected_model))

        if processed is not None:
            plot_feature_importance(all_models_full.get("rf"), processed)
            plot_pie_chart(max(score,0))
            plot_radar(model_scores)
            plot_boxplot(processed)
            plot_correlation_heatmap(df)
            download_model_report(processed)

            try:
                module_map = {"💳 Credit Card":"creditcard","📱 PaySim":"paysim","🏦 Loan":"loan","🚗 Insurance":"insurance"}
                plot_permutation_importance(module_map[selected_tab])
            except:
                pass
