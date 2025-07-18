import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd

# Bar chart of model predictions
def plot_bar(scores):
    st.subheader("📊 Model Scores")
    fig, ax = plt.subplots()
    names = list(scores.keys())
    values = list(scores.values())
    sns.barplot(x=names, y=values, ax=ax, palette="viridis")
    ax.set_ylabel("Fraud Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# Heatmap of input features
def plot_heatmap(df):
    st.subheader("🧯 Feature Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f", ax=ax)
    st.pyplot(fig)

# SHAP summary plot
def plot_shap_summary(model, X):
    st.subheader("🔎 SHAP Explanation")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    fig = shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(bbox_inches="tight")
