

import streamlit as st
import pandas as pd
import time
from fraud_modules import credit_card, paysim, loan, insurance
from utils.visualizer import plot_bar, plot_shap_summary
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="🛡️ Multi-Fraud Detector", layout="wide", page_icon="💳")

mode = st.sidebar.selectbox("🎨 Theme Mode", ["Dark Mode", "Light Mode"])
bg_color = "#1b2735" if mode == "Dark Mode" else "#f0f2f6"
font_color = "#ffffff" if mode == "Dark Mode" else "#111111"

# Reset Button
if st.sidebar.button("🔁 Reset App"):
    st.session_state.clear()
    st.rerun()

# Sidebar Model Info
if "Model Info" not in st.session_state:
    st.session_state["Model Info"] = {
        "💳 Credit Card": "RandomForest + XGBoost + CatBoost | ~99% Accuracy",
        "📱 PaySim": "IsolationForest + LogisticRegression | Balanced Recall",
        "🏦 Loan": "LightGBM + LogisticRegression | Handles missing data",
        "🚗 Insurance": "CatBoost + Ensemble | Imbalanced friendly"
    }

# Show model explanations
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Explanation")
model_descriptions = st.session_state["Model Info"]
for name, desc in model_descriptions.items():
    with st.sidebar.expander(name):
        st.markdown(desc)

# Starfield CSS + canvas
st.markdown(f"""
    <style>
        body {{ background: linear-gradient(-45deg, {bg_color}, #203a43, #2c5364); background-size: 400% 400%; animation: gradientBG 20s ease infinite; color: {font_color}; }}
        @keyframes gradientBG {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}
        .block-container {{ backdrop-filter: blur(6px); background-color: rgba(0,0,0,0.3); padding: 2rem; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.2); }}
    </style>
""", unsafe_allow_html=True)

st.components.v1.html("""
<canvas id=\"stars\" style=\"position:fixed; top:0; left:0; z-index:-1;\"></canvas>
<script>
var canvas = document.getElementById('stars');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
var ctx = canvas.getContext('2d');
var stars = [], FPS = 60;
for (var i = 0; i < 250; i++) {
  stars.push({x: Math.random()*canvas.width,y: Math.random()*canvas.height,radius: Math.random()*1.2,vx: -0.5+Math.random(),vy: -0.5+Math.random()});
}
function draw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = 'white';
  ctx.shadowBlur = 3;
  ctx.shadowColor = 'white';
  for (var i = 0; i < stars.length; i++) {
    var s = stars[i];
    ctx.beginPath();ctx.arc(s.x,s.y,s.radius,0,2*Math.PI);ctx.fill();
  }
  move();
}
function move() {
  for (var b = 0; b < stars.length; b++) {
    var s = stars[b];
    s.x += s.vx;
    s.y += s.vy;
    if (s.x < 0 || s.x > canvas.width) s.vx = -s.vx;
    if (s.y < 0 || s.y > canvas.height) s.vy = -s.vy;
  }
}
setInterval(draw, 1000/FPS);
</script>
""", height=0)

# Fraud Tab Routing
fraud_modules = {
    "💳 Credit Card": credit_card,
    "📱 PaySim": paysim,
    "🏦 Loan": loan,
    "🚗 Insurance": insurance
}
tabs = ["🏠 Home"] + list(fraud_modules.keys())
selected_tab = st.sidebar.radio("Select Fraud Type:", tabs)

if selected_tab == "🏠 Home":
    st.title("🛡️ Multi-Fraud Detection System")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💳 Credit Card Fraud"): st.session_state["page"] = "💳 Credit Card"; st.experimental_rerun()
        if st.button("🏦 Loan Fraud"): st.session_state["page"] = "🏦 Loan"; st.experimental_rerun()
    with col2:
        if st.button("📱 PaySim Fraud"): st.session_state["page"] = "📱 PaySim"; st.experimental_rerun()
        if st.button("🚗 Insurance Fraud"): st.session_state["page"] = "🚗 Insurance"; st.experimental_rerun()

if selected_tab in fraud_modules:
    st.title(f"{selected_tab} Fraud Detection")
    uploaded = st.file_uploader("📥 Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        if st.button("🔍 Predict Fraud"):
            st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
            with st.spinner("Analyzing with 6 AI models..."):
                time.sleep(1)
                fn = f"predict_{selected_tab.lower().split()[0]}_fraud"
                score, model_scores, processed = fraud_modules[selected_tab].__getattribute__(fn)(df)

            st.success(f"🧠 Final Score: {score*100:.2f}% Fraud Likely")
            plot_bar(model_scores)
            plot_shap_summary(fraud_modules[selected_tab].models['rf'], processed)

            # Export result
            if st.button("⬇️ Download Results"):
                result_df = df.copy()
                result_df['FraudScore'] = score
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download as CSV", data=csv, file_name="fraud_results.csv")

            # Scoreboard
            st.markdown("### 📊 Scoreboard")
            st.bar_chart(pd.DataFrame.from_dict(model_scores, orient='index', columns=['Score']))

            # Confusion Matrix (only if ground truth present)
            if 'actual' in df.columns:
                y_true = df['actual']
                y_pred = [1 if model_scores['rf'] > 0.5 else 0]*len(df)
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

# Placeholder for future chatbot
with st.sidebar.expander("💬 Assistant (Coming Soon)"):
    st.info("Ask me anything about fraud detection.")
    st.text_input("💡 Example: How does the model detect anomalies?")
