import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import io
import base64

# ------------------------------
# 📊 All Model Scores Bar Chart
# ------------------------------
def plot_bar(model_scores):
    st.subheader("📊 All Model Prediction Scores")
    df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['Score']).sort_values(by='Score', ascending=False)
    st.bar_chart(df)

    selected_model = st.selectbox("🔍 Select a Model to Inspect", df.index.tolist(), key="model_inspector")
    if selected_model:
        st.markdown(f"### 🔎 {selected_model.upper()} Model Insights")
        st.metric("Fraud Confidence Score", f"{model_scores[selected_model]*100:.2f}%")
        st.bar_chart(pd.DataFrame([model_scores[selected_model]], index=[selected_model], columns=["Score"]))
        st.markdown(get_model_description(selected_model))

# ------------------------------
# 🧠 SHAP Summary / Waterfall / Force
# ------------------------------
def plot_shap_summary(model, X_processed):
    st.subheader("🧠 SHAP Explanation")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)

        if isinstance(shap_values, list):  # Multi-class
            shap_values = shap_values[0]

        if len(X_processed) < 2:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — showing single row waterfall plot")
            try:
                exp = shap.Explanation(values=shap_values[0],
                                       base_values=explainer.expected_value,
                                       data=X_processed.iloc[0])
                shap.plots.waterfall(exp, show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"❌ Waterfall plot failed: {e}")
        else:
            try:
                shap.summary_plot(shap_values, X_processed, show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"❌ SHAP summary plot failed: {e}")
    except Exception as e:
        st.error(f"❌ SHAP explanation failed: {e}")

# ------------------------------
# 🧬 SHAP Force Plot (for one row)
# ------------------------------
def plot_shap_force(model, X_processed):
    st.subheader("🧬 SHAP Force Plot (Row 0)")
    try:
        explainer = shap.Explainer(model, X_processed)
        shap_values = explainer(X_processed)
        force_html = shap.plots.force(shap_values[0], matplotlib=False)
        b64 = base64.b64encode(force_html.data.encode()).decode()
        html = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="400"></iframe>'
        st.components.v1.html(html, height=400)
    except Exception as e:
        st.error(f"❌ SHAP force plot failed: {e}")

# ------------------------------
# 🥧 Enhanced Pie Chart (Tiny Safe)
# ------------------------------
def plot_pie_chart(probability_score):
    st.subheader("🥧 Estimated Fraud Likelihood")
    values = [probability_score, 1 - probability_score]
    labels = ['Fraud', 'Not Fraud']
    explode = [0.1 if v < 0.01 else 0 for v in values]

    def fmt(pct):
        return f"{pct:.5f}%" if pct < 0.01 else f"{pct:.1f}%"

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(values, labels=labels, explode=explode,
                                      autopct=lambda pct: fmt(pct), startangle=90,
                                      colors=['#ff6b6b', '#51cf66'])
    ax.axis('equal')
    st.pyplot(fig)

# ------------------------------
# 📋 Confusion Matrix and Report
# ------------------------------
def plot_confusion_report(y_true, y_pred):
    st.subheader("📋 Model Evaluation Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df.style.highlight_max(axis=0))

    st.markdown("#### 🔢 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ------------------------------
# 📦 Box Plot of Features
# ------------------------------
def plot_boxplot(df):
    st.subheader("📦 Feature Distribution – Box Plot")
    if df.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.info("📭 No numeric data to plot.")

# ------------------------------
# 🕸 Radar Chart for Model Scores
# ------------------------------
def plot_radar(model_scores):
    st.subheader("🕸 Radar Chart – Model Comparison")
    labels = list(model_scores.keys())
    values = list(model_scores.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ------------------------------
# 🌡️ Correlation Heatmap (Optional)
# ------------------------------
def plot_correlation_heatmap(df):
    st.subheader("🌡️ Correlation Heatmap")
    if df.shape[1] > 1:
        corr = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough features for correlation matrix.")

# ------------------------------
# 🖨️ Download CSV Report
# ------------------------------
def download_model_report(df, filename="fraud_report.csv"):
    st.download_button("⬇️ Download Model Report CSV", data=df.to_csv(index=False).encode(), file_name=filename, mime="text/csv")

# ------------------------------
# 🧾 Model Descriptions
# ------------------------------
def get_model_description(model_key):
    descriptions = {
        "rf": "🌲 **Random Forest**: Ensemble of decision trees. Great for general tabular prediction tasks.",
        "xgb": "🚀 **XGBoost**: Powerful gradient boosting model. Fast, accurate, and handles overfitting well.",
        "lgbm": "🔆 **LightGBM**: Light-weight boosting model. Great for large datasets and speed.",
        "cat": "🐱 **CatBoost**: From Yandex. Excellent handling of categorical variables and missing values.",
        "lr": "📐 **Logistic Regression**: A simple, interpretable linear model for binary classification.",
        "iso": "🚨 **Isolation Forest**: Anomaly detection technique for unsupervised fraud flagging."
    }
    return descriptions.get(model_key, "ℹ️ No detailed info for this model.")
