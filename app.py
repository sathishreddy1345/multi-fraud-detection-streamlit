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
        print("❌ Could not load metrics:", e)
        return {}


# Page setup
st.set_page_config(page_title="🛡️ Multi-Fraud Detection System", layout="wide", page_icon="🧠")

# 🎨 Styling
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

# --------------------------
# Session Management for Tab
# --------------------------
tabs = ["🏠 Home", "💳 Credit Card", "📱 PaySim", "🏦 Loan", "🚗 Insurance"]

if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = "🏠 Home"

selected_tab = st.sidebar.radio("Select Fraud Type", tabs, index=tabs.index(st.session_state["selected_tab"]))

# --------------------------
# Sidebar Info & Reset
# --------------------------
st.sidebar.title("🧭 Navigation Panel")

if "Model Info" not in st.session_state:
    st.session_state["Model Info"] = {
        "💳 Credit Card": "RandomForest, XGBoost, CatBoost, Logistic Regression, IsolationForest",
        "📱 PaySim": "Logistic Regression + IsolationForest",
        "🏦 Loan": "XGBOOST + Logistic Regression",
        "🚗 Insurance": "CatBoost + Random Forest"
    }

with st.sidebar.expander("📘 Model Details", expanded=False):
    for model, desc in st.session_state["Model Info"].items():
        st.markdown(f"**{model}**: {desc}")

if st.sidebar.button("🔁 Reset App"):
    st.session_state.clear()
    st.experimental_rerun()

# --------------------------
# Fraud Module Mapping
# --------------------------
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

# --------------------------
# Home Page
# --------------------------
if selected_tab == "🏠 Home":
    st.title("🛡️ Multi-Fraud Detection Dashboard")
    st.markdown("Choose a fraud type from the sidebar or buttons below to start detection.")

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

# --------------------------
# Prediction Pages
# --------------------------
if selected_tab in fraud_modules:
    st.title(f"{selected_tab} Detection")
    uploaded = st.file_uploader("📤 Upload a CSV file for analysis", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded, thousands=",")
        # 🧼 Force numeric conversion for all columns
        for col in df.columns:
            if df[col].dtype == object:
                # Try to clean numeric-looking strings
                cleaned = df[col].astype(str).str.replace(",", "").str.replace("₹", "")
                try:
                    df[col] = pd.to_numeric(cleaned)
                except:
                    df[col] = cleaned  # keep as string if not numeric


        # Drop any remaining non-numeric or fully empty columns
        df = df.dropna(axis=1, how="all")


        edited_df = st.data_editor(df.head(50), use_container_width=True, num_rows="dynamic", key="editor_input")

        if st.button("🔍 Run AI Fraud Detection"):
            df = edited_df
            with st.spinner("Analyzing... please wait ⏳"):
                time.sleep(1)
                fn = function_map[selected_tab]
                try:
                    score, model_scores, processed = getattr(fraud_modules[selected_tab], fn)(df)
                    st.session_state["model_scores"] = model_scores
                    st.session_state["score"] = score
                    st.session_state["processed"] = processed
                    st.session_state["uploaded_df"] = df
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    st.error(f"❌ Prediction failed: {e}")
                    st.stop()

    # Display results if available
    if "model_scores" in st.session_state and st.session_state["selected_tab"] == selected_tab:
        model_scores = st.session_state["model_scores"]
        score = st.session_state["score"]
        processed = st.session_state["processed"]
        df = st.session_state["uploaded_df"]

        if not model_scores:
            st.error("❌ No models were able to make predictions.")
        else:
            selected_model = plot_bar(model_scores, key=f"{selected_tab}_bar")
                    # ============================================================
            # 🔥 Research-Grade Ensemble (Soft Voting, No Normalization)
            # ============================================================
            
            import numpy as np
            
            st.markdown("## 🔍 Research Ensemble Score (Soft Voting)")
            
            model_list = list(model_scores.keys())
            scores_array = np.array(list(model_scores.values()))
            
            # Equal weights for all models (standard)
            weights = np.ones(len(model_list))
            
            # Weighted soft vote
            ensemble_research = np.sum(weights * scores_array) / np.sum(weights)
            
            st.metric("📌 Research Ensemble Fraud Likelihood", f"{ensemble_research * 100:.2f}%")
            
            # Table for paper
            df_w = pd.DataFrame({
                "Model": model_list,
                "Prediction Score": scores_array,
                "Weight": weights
            })
            
            st.markdown("### 📑 Ensemble Table (Use in Research Paper)")
            st.dataframe(df_w.style.format({"Prediction Score": "{:.4f}", "Weight": "{:.2f}"}))



                        # ============================================================
            # 📊 Full Research Evaluation Block (All Metrics)
            # ============================================================
            
            st.markdown("---")
            st.markdown("## 🧪 Research Evaluation Metrics")
            
            if "actual" not in processed.columns:
                st.warning("⚠️ Ground truth labels ('actual') were not found. Metrics cannot be computed.")
            else:
                from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix
            )
            
            # FIX: Force numeric ground truth
                y_true = processed["actual"].astype(int).to_numpy()
                
                # FIX: Expand ensemble prediction and force int dtype
                y_pred_ensemble = np.array(
                    [1 if ensemble_research > 0.5 else 0] * len(y_true),
                    dtype=int
                )
                
                # FIX: Guarantee equal shape
                if len(y_true) != len(y_pred_ensemble):
                    raise ValueError("Prediction and ground truth length mismatch.")
                
                metrics_ensemble = {
                    "Accuracy": accuracy_score(y_true, y_pred_ensemble),
                    "Precision": precision_score(y_true, y_pred_ensemble, zero_division=0),
                    "Recall": recall_score(y_true, y_pred_ensemble, zero_division=0),
                    "F1 Score": f1_score(y_true, y_pred_ensemble, zero_division=0),
                }
                
                # AUC
                try:
                    metrics_ensemble["AUC-ROC"] = roc_auc_score(
                        y_true,
                        np.array([ensemble_research] * len(y_true), dtype=float)
                    )
                except:
                    metrics_ensemble["AUC-ROC"] = "N/A"
                
                st.json(metrics_ensemble)

            
              
                # ---------------------------------------------------------
                # 📊 Confusion Matrix for Ensemble
                # ---------------------------------------------------------
                st.markdown("### 📉 Ensemble Confusion Matrix")
            
                cm = confusion_matrix(y_true, y_pred_ensemble)
                cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            
                st.dataframe(cm_df.style.background_gradient(cmap="rocket_r"))


                metrics = load_metrics()

                if not metrics:
                    st.error("Metrics file not found. Please upload retrained metrics JSON.")
                else:
                    st.subheader("📊 Credit Card Fraud Model Metrics")
                
                    model_list = {
                        "Random Forest": "rf",
                        "XGBoost": "xgb",
                        "CatBoost": "cat",
                        "Logistic Regression": "lr",
                        "IsolationForest": "iso"
                    }
                
                    selected_model_name = st.selectbox("Select Model", list(model_list.keys()))
                    key = model_list[selected_model_name]
                
                    model_metrics = metrics.get(key, None)
                
                    if not model_metrics:
                        st.error(f"No metrics found for model: {selected_model_name}")
                    else:
                        st.success(f"Showing metrics for: {selected_model_name}")
                
                        st.markdown("### 🧮 Performance Metrics")
                
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
                        col2.metric("Precision", f"{model_metrics['precision']:.4f}")
                        col3.metric("Recall", f"{model_metrics['recall']:.4f}")
                
                        col4, col5 = st.columns(2)
                        col4.metric("F1 Score", f"{model_metrics['f1']:.4f}")
                
                        roc = model_metrics.get("roc_auc")
                        if roc is not None:
                            col5.metric("ROC-AUC", f"{roc:.4f}")
                        else:
                            col5.metric("ROC-AUC", "N/A")
                
                        # Confusion Matrix
                        st.markdown("### 🔲 Confusion Matrix")
                        cm = model_metrics["confusion_matrix"]
                        st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]))
                
                        # Classification Report
                        st.markdown("### 📄 Classification Report")
                        st.text(model_metrics["classification_report"])

            
                # ---------------------------------------------------------
                # 📑 Downloadable Table for Research Paper
                # ---------------------------------------------------------
                st.markdown("### 📥 Download Research Metrics Table")
                
                final_table = df_model_metrics.copy()
                final_table.loc[len(final_table)] = ["Ensemble"] + list(metrics_ensemble.values())
            
                st.dataframe(final_table.style.format("{:.4f}"))
            
                csv = final_table.to_csv(index=False)
                st.download_button("📄 Download Metrics as CSV", csv, "research_metrics.csv", "text/csv")


            


            if selected_model is None:
                selected_model = next(iter(model_scores))  # fallback

            all_models = fraud_modules[selected_tab].models
            default_model = all_models.get("rf") or list(all_models.values())[0]
            all_models_full = fraud_modules[selected_tab].models_full
            default_model_full = all_models_full.get("rf") or list(all_models_full.values())[0]


            if processed is not None and not processed.isnull().all().all():
                plot_feature_importance(default_model_full, processed)

                plot_pie_chart(max(score, 0))
                st.success(f"✅ Overall Fraud Likelihood: **{score * 100:.2f}%**")

                st.markdown("### 🔬 Explore Individual Model")
                selected_model = st.selectbox(
                    "Choose a model",
                    list(model_scores.keys()),
                    index=0,
                    key=f"model_inspector_{selected_tab}"
                )
                st.metric("Score", f"{model_scores[selected_model]*100:.2f}%")
                st.markdown(get_model_description(selected_model))

                # Confusion Matrix
                if 'actual' in df.columns:
                    y_true = df['actual']
                    # Ensemble prediction for each row (same value repeated)
                    y_pred_ensemble = np.array([1 if ensemble_research > 0.5 else 0] * len(y_true))

                    plot_confusion_report(y_true, y_pred)

                plot_radar(model_scores)
                plot_boxplot(processed)


                plot_correlation_heatmap(df)
                download_model_report(processed)

                try:
                    model_object = all_models[selected_model]
                    y_true = df['actual'] if 'actual' in df.columns else None
                    module_key_map = {
                        "💳 Credit Card": "creditcard",
                        "📱 PaySim": "paysim",
                        "🏦 Loan": "loan",
                        "🚗 Insurance": "insurance"
                                   }
                    module_name = module_key_map[selected_tab]
                    plot_permutation_importance(module_name)


                except Exception as e:
                    st.warning(f"⚠️ Permutation importance failed: {e}")
