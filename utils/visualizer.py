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
    st.subheader("🧠 SHAP Explanation")

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        if len(X) < 2:
            st.warning("⚠️ SHAP beeswarm needs ≥2 rows — using waterfall plot for row 0.")

            try:
                # Check if SHAP values are multi-class (3D shape: samples × classes × features)
                if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                    st.info("Multi-class model detected — showing class 0 explanation.")

                    # Create a proper SHAP Explanation for the first row, first class
                    single_explanation = shap.Explanation(
                        values=shap_values.values[0, 0],
                        base_values=shap_values.base_values[0, 0],
                        data=shap_values.data[0],
                        feature_names=shap_values.feature_names
                    )
                    shap.plots.waterfall(single_explanation, show=False)

                else:
                    shap.plots.waterfall(shap_values[0], show=False)

                st.pyplot(plt.gcf())

            except Exception as e:
                st.error(f"❌ Waterfall plot failed:\n\n{str(e)}")

        else:
            try:
                shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"❌ SHAP beeswarm failed:\n\n{str(e)}")

    except Exception as e:
        st.error(f"❌ SHAP explainer error:\n\n{str(e)}")
