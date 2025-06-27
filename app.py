import streamlit as st
import pandas as pd
import time
import streamlit.components.v1 as components
from fraud_modules import credit_card
from fraud_modules import paysim
from fraud_modules import loan
from fraud_modules import vehicle_loan
from utils.visualizer import plot_bar, plot_heatmap, plot_shap_summary

st.set_page_config(page_title="🛡️ Multi-Fraud Detection Dashboard", layout="wide", page_icon="💳")

with st.sidebar:
    st.title("🧠 Fraud Detector")
    st.markdown("Built with 💼 Machine Learning")
    st.markdown("Select fraud type to test your model")
    st.info("Upload CSV file → Get fraud probability → Visualize")

st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Multi-Fraud Detection Dashboard")

fraud_tabs = st.tabs(["💳 Credit Card", "📱 PaySim", "🏦 Loan", "🚗 Vehicle Loan"])

with fraud_tabs[0]:
    st.header("💳 Credit Card Fraud Detection")
    uploaded_file = st.file_uploader("Upload Credit Card Transaction CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🔍 Predict Fraud"):
            with st.spinner("Analyzing with 6 AI models..."):
                time.sleep(2)
                combined, scores, processed_df = credit_card.predict_creditcard_fraud(df)

            st.success(f"🧠 Final Fraud Risk Score: {combined*100:.2f}%")

            if combined > 0.5:
                st.error("🚨 High risk of fraud detected!")
                st.balloons()
            else:
                st.success("✅ Low risk of fraud.")
                st.snow()

            plot_bar(scores)
            plot_heatmap(df)
            # SHAP visualization using RF model only for speed
            from fraud_modules.credit_card import models
            if 'rf' in models:
                plot_shap_summary(models['rf'], processed_df)

            st.markdown("---")
            st.caption("🔎 Powered by 6 AI Models | Real-time Risk Estimation + SHAP Insights")
           
with fraud_tabs[1]:
    st.header("📱 Mobile Transaction Fraud (PaySim)")
    uploaded_file = st.file_uploader("Upload PaySim Dataset CSV", type=["csv"], key="paysim")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🔍 Predict PaySim Fraud"):
            with st.spinner("Running models on mobile transaction fraud data..."):
                time.sleep(2)
                combined, scores, processed_df = paysim.predict_paysim_fraud(df)

            st.success(f"📱 Fraud Score: {combined*100:.2f}%")

            if combined > 0.5:
                st.error("🚨 Fraud likely in PaySim dataset")
                st.balloons()
            else:
                st.success("✅ Transactions appear safe.")
                st.snow()

            plot_bar(scores)
            plot_heatmap(df)
            from fraud_modules.paysim import models as paysim_models
            if 'rf' in paysim_models:
                plot_shap_summary(paysim_models['rf'], processed_df)


with fraud_tabs[2]:
    st.header("🏦 Loan Application Fraud")
    uploaded_file = st.file_uploader("Upload Loan Dataset CSV", type=["csv"], key="loan")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🔍 Predict Loan Fraud"):
            with st.spinner("Running loan fraud prediction models..."):
                time.sleep(2)
                combined, scores, processed_df = loan.predict_loan_fraud(df)

            st.success(f"🏦 Fraud Score: {combined*100:.2f}%")

            if combined > 0.5:
                st.error("🚨 Potential loan fraud detected")
                st.balloons()
            else:
                st.success("✅ Loan application appears safe.")
                st.snow()

            plot_bar(scores)
            plot_heatmap(df)
            from fraud_modules.loan import models as loan_models
            if 'rf' in loan_models:
                plot_shap_summary(loan_models['rf'], processed_df)


with fraud_tabs[3]:
    st.header("🚗 Vehicle Loan Fraud")
    uploaded_file = st.file_uploader("Upload Vehicle Loan Dataset CSV", type=["csv"], key="vehicle")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🔍 Predict Vehicle Loan Fraud"):
            with st.spinner("Analyzing vehicle loan fraud risk..."):
                time.sleep(2)
                combined, scores, processed_df = vehicle_loan.predict_vehicle_loan_fraud(df)

            st.success(f"🚗 Fraud Score: {combined*100:.2f}%")

            if combined > 0.5:
                st.error("🚨 Suspicious vehicle loan detected")
                st.balloons()
            else:
                st.success("✅ No signs of vehicle loan fraud.")
                st.snow()

            plot_bar(scores)
            plot_heatmap(df)
            from fraud_modules.vehicle_loan import models as vehicle_models
            if 'rf' in vehicle_models:
                plot_shap_summary(vehicle_models['rf'], processed_df)


 st.markdown("---")
            st.caption("🔎 Powered by 6 AI Models | Real-time Risk Estimation + SHAP Insights")
           

