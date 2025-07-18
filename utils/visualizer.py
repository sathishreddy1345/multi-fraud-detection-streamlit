import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap

# 📊 Bar chart of model scores
def plot_bar(model_scores):
    if not model_scores:
        st.warning("⚠️ No model scores to display.")
        return
    fig, ax = plt.subplots()
    ax.barh(list(model_scores.keys()), list(model_scores.values()), color='skyblue')
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_title("🔍 Model Predictions")
    st.pyplot(fig)

# 🔎 SHAP summary/waterfall plots
def plot_shap_summary(model, X):
    st.subheader("🔍 SHAP Explanation (RandomForest or any ML model)")
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        if len(X) < 2:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — using waterfall plot for row 0.")
            try:
                # For multi-output models (like multiclass), use class 0 if needed
                if hasattr(shap_values[0], "values"):
                    values = shap_values[0].values
                    if len(values.shape) > 1:
                        # Multi-class — use first class
                        fig = shap.plots.waterfall(shap_values[0, 0], show=False)
                    else:
                        fig = shap.plots.waterfall(shap_values[0], show=False)
                else:
                    fig = shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"❌ Waterfall plot failed:\n\n{str(e)}")
        else:
            try:
                fig = shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"❌ SHAP summary plot failed:\n\n{str(e)}")
    except Exception as e:
        st.error(f"❌ SHAP explainer error:\n\n{str(e)}")
