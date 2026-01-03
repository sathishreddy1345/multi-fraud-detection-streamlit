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


# -------------------- LOAD METRICS --------------------
def load_metrics():
    try:
        with open("models/credit_card_metrics.json") as f:
            return json.load(f)
    except Exception as e:
        print("❌ Could not load metrics:", e)
        return {}


# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="🛡️ Multi-Fraud Detection System",
    layout="wide",
    page_icon="🧠"
)


# -------------------- STYLING --------------------
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


# -------------------- TABS --------------------
tabs = ["🏠 Home", "💳 Credit Card", "📱 PaySim", "🏦 Loan", "🚗 Insurance"]

if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = "🏠 Home"

selected_tab = st.sidebar.radio(
    "Select Fraud Type",
    tabs,
    index=tabs.index(st.session_state["selected_tab"])
)


# -------------------- SIDEBAR --------------------
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


# -------------------- MODULE MAPS --------------------
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


# -------------------- HOME PAGE --------------------
if selected_tab == "🏠 Home":
    st.title("🛡️ Multi-Fraud Detection Dashboard")
    st.markdown("Choose a fraud type from the sidebar or buttons below.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💳 Credit Card Fraud"):
            st.session_state["selected_tab"] = "💳 Credit Card"
            st.experimental_rerun()
        if st.button("🏦 Loan Fraud"):
            st.session_state["selected_tab"] = "🏦 Loan"
            st.experimental_rerun()
    with col2:
        if st.button("📱 PaySim Fraud"):
            st.session_state["selected_tab"] = "📱 PaySim"
            st.experimental_rerun()
        if st.button("🚗 Insurance Fraud"):
            st.session_state["selected_tab"] = "🚗 Insurance"
            st.experimental_rerun()


# =====================================================
#                 P R E D I C T I O N   P A G E S
# =====================================================
if selected_tab in fraud_modules:

    module = fraud_modules[selected_tab]
    st.title(f"{selected_tab} Detection")

    df = None

    # ---------- Input Options ----------
    input_mode = st.radio(
        "Choose Input Method",
        ["📤 Upload CSV File", "🧾 Use Auto-Template (Fill Values)"],
        horizontal=True
    )

    # ---------- MODE 1: Upload ----------
    if input_mode == "📤 Upload CSV File":
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded, thousands=",")
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")

            df = st.data_editor(
                df.head(50),
                use_container_width=True,
                num_rows="dynamic",
                key="upload_editor",
                column_config={
                    col: st.column_config.NumberColumn(step=0.01)
                    for col in df.columns
                }
            )

    # ---------- MODE 2: Auto Template ----------
    if input_mode == "🧾 Use Auto-Template (Fill Values)":
        if hasattr(module, "get_template_df"):
            tmpl = module.get_template_df().iloc[:1].copy()

            st.info("Enter values in this row. Only one record is allowed.")

            for c in tmpl.columns:
                tmpl[c] = tmpl[c].astype(float)

            df = st.data_editor(
                tmpl,
                use_container_width=True,
                num_rows=1,
                key=f"{selected_tab}_template_editor",
                column_config={
                    col: st.column_config.NumberColumn(step=0.01)
                    for col in tmpl.columns
                }
            )
        else:
            st.error("Template not available for this module.")

    # ---------- CLEAN → RUN ----------
    if df is not None:

        for col in df.columns:
            cleaned = (
                df[col].astype(str)
                .str.replace(",", "")
                .str.replace("₹", "")
                .str.strip()
            )
            try:
                df[col] = cleaned.astype(float)
            except:
                pass

        df = df.dropna(axis=1, how="all")

        if st.button("🔍 Run AI Fraud Detection"):
            with st.spinner("Analyzing..."):
                try:
                    fn = function_map[selected_tab]
                    score, model_scores, processed = getattr(module, fn)(df)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                st.session_state["model_scores"] = model_scores
                st.session_state["score"] = score
                st.session_state["processed"] = processed
                st.session_state["uploaded_df"] = df


# =====================================================
#           R E S U L T S   &   V I S U A L S
# =====================================================
if "model_scores" in st.session_state and st.session_state["selected_tab"] == selected_tab:

    model_scores = st.session_state["model_scores"]
    score = st.session_state["score"]
    processed = st.session_state["processed"]
    df = st.session_state["uploaded_df"]

    # ---------- Model Scores ----------
    st.markdown("## 📊 Individual Model Prediction Scores")
    selected_model = plot_bar(model_scores, key=f"{selected_tab}_bar")

    # ---------- Ensemble ----------
    st.markdown("## 🔥 Optimized Ensemble Score (F1-Weighted Soft Voting)")
    metrics = load_metrics()

    if metrics:
        model_list = list(model_scores.keys())
        scores_array = np.array([model_scores[m] for m in model_list])
        weights = np.array([metrics[m]["f1"] if m in metrics else 1 for m in model_list])
        ensemble_weighted = np.sum(weights * scores_array) / np.sum(weights)

        st.metric("📌 Ensemble Fraud Likelihood", f"{ensemble_weighted * 100:.2f}%")

    # ---------- Best Model ----------
    if metrics:
        st.markdown("## 🏆 Best Performing Model")
        best_key = max(metrics.keys(), key=lambda m: metrics[m]["f1"])
        name_map = {
            "rf": "Random Forest",
            "xgb": "XGBoost",
            "cat": "CatBoost",
            "lr": "Logistic Regression",
            "iso": "IsolationForest"
        }
        st.success(
            f"⭐ Best Model: **{name_map[best_key]}** — "
            f"F1 Score: **{metrics[best_key]['f1']:.4f}**"
        )

    st.markdown("---")

    # ---------- Metrics ----------
    if metrics:
        st.markdown("## 📊 Model Evaluation Metrics")
        display_names = {
            "Random Forest": "rf",
            "XGBoost": "xgb",
            "CatBoost": "cat",
            "Logistic Regression": "lr",
            "IsolationForest": "iso"
        }
        choice = st.selectbox("🔽 Select Model", list(display_names.keys()))
        key = display_names[choice]
        m = metrics[key]

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{m['accuracy']:.4f}")
        col2.metric("Precision", f"{m['precision']:.4f}")
        col3.metric("Recall", f"{m['recall']:.4f}")

        col4, col5 = st.columns(2)
        col4.metric("F1 Score", f"{m['f1']:.4f}")
        col5.metric("ROC-AUC", f"{m['roc_auc']:.4f}" if m["roc_auc"] else "N/A")

    st.markdown("---")

    # ---------- Model Inspection ----------
    st.success(f"🎯 Overall Fraud Likelihood: **{score*100:.2f}%**")

    st.markdown("### 🔬 Explore Individual Model")
    inspect_model = st.selectbox("Choose model", list(model_scores.keys()))
    st.metric("Model Score", f"{model_scores[inspect_model]*100:.2f}%")

    if processed is not None:
        all_models_full = fraud_modules[selected_tab].models_full

        if "rf" in all_models_full:
            plot_feature_importance(all_models_full["rf"], processed)

        plot_pie_chart(max(score, 0))
        plot_radar(model_scores)
        plot_boxplot(processed)
        plot_correlation_heatmap(df)
        download_model_report(processed)
