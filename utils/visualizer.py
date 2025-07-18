# utils/visualizer.py

import streamlit as st
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 📊 Overall bar chart for all model scores
def plot_bar(model_scores):
    st.subheader("📊 Model Prediction Confidence")
    df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['Score'])
    df = df.sort_values(by='Score', ascending=False)
    st.bar_chart(df)

# 🔍 Detailed model insights
def plot_model_insight(model_name, model, X_processed):
    st.markdown(f"## 🔎 Insights for `{model_name.upper()}` model")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_names = X_processed.columns if isinstance(X_processed, pd.DataFrame) else [f'Feature {i}' for i in range(len(importances))]
        sorted_idx = np.argsort(importances)[-10:]

        st.write("### 🔥 Top Features by Importance")
        fig, ax = plt.subplots()
        sns.barplot(x=importances[sorted_idx], y=np.array(feat_names)[sorted_idx], ax=ax)
        st.pyplot(fig)

    if model_name != 'iso':  # Skip SHAP for Isolation Forest
        try:
            explainer = shap.Explainer(model, X_processed)
            shap_values = explainer(X_processed)

            st.write("### 🧠 SHAP Explanation")
            if len(X_processed) < 2:
                st.warning("SHAP needs ≥2 rows — showing waterfall plot for row 0.")
                try:
                    fig = shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(bbox_inches='tight')
                except Exception as e:
                    st.error(f"❌ Waterfall plot failed:\n\n{e}")
            else:
                fig = shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(bbox_inches='tight')

        except Exception as e:
            st.error(f"❌ SHAP plot failed:\n\n{e}")


def plot_shap_summary(model, X_processed):
    st.markdown("### 🧠 SHAP Explanation")

    try:
        explainer = shap.Explainer(model, X_processed)
        shap_values = explainer(X_processed)

        if len(X_processed) < 2:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — using waterfall plot for row 0.")

            try:
                fig = shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"❌ Waterfall plot failed:\n\n{e}")
        else:
            try:
                fig = shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"❌ SHAP summary plot failed:\n\n{e}")

    except Exception as e:
        st.error(f"❌ SHAP explanation failed:\n\n{e}")
