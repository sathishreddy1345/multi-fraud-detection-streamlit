
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_bar(scores: dict):
    st.subheader("📊 Individual Model Confidence")
    bar_df = pd.DataFrame(scores.items(), columns=["Model", "Score"])
    fig, ax = plt.subplots()
    sns.barplot(data=bar_df, x="Model", y="Score", ax=ax, palette="viridis")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

def plot_heatmap(data):
    st.subheader("🔥 Feature Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)
