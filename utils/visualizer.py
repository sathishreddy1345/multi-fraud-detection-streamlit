import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# ------------------------------
# 📥 Load dataset for module
# ------------------------------
def load_dataset_for_module(module_name):
    try:
        module_map = {
            "loan": "data/loan.csv",
            "credit_card": "data/creditcard.csv",
            "paysim": "data/paysim.csv",
            "insurance": "data/insurance.csv"
        }
        path = module_map.get(module_name.lower())
        if not path:
            st.warning(f"⚠️ No dataset file mapped for: {module_name}")
            return None

        df = pd.read_csv(path)
        df = df.select_dtypes(include=[np.number]).fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset for module `{module_name}`: {e}")
        return None

# ------------------------------
# 📊 Bar Chart
# ------------------------------
def plot_bar(model_scores, key=None):
    st.subheader("📊 All Model Prediction Scores")
    df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['Score']).sort_values(by='Score', ascending=False)

    if df.empty:
        st.warning("⚠️ No model scores to display.")
        return None

    st.bar_chart(df)

    selected_model = st.selectbox("🔍 Select a Model to Inspect", df.index.tolist(), key=key)
    if selected_model:
        st.metric("Fraud Confidence Score", f"{model_scores[selected_model]*100:.2f}%")
        st.markdown(get_model_description(selected_model))
    return selected_model

# ------------------------------
# 🔍 Feature Importance Plot
# ------------------------------
def plot_feature_importance(model_tuple, X_processed):
    st.subheader("📌 Feature Importance (Model-Based)")

    # Inside plot_feature_importance
    if isinstance(model_tuple, tuple):
        model, feature_columns = model_tuple
    else:
        model = model_tuple
        feature_columns = X_processed.columns.tolist()


    try:
        model, feature_columns = model_tuple

        if hasattr(model, "feature_importances_") and feature_columns is not None:
            # Filter only trained columns
            X_features = X_processed[feature_columns]

            importances = model.feature_importances_
            if len(importances) != len(feature_columns):
                raise ValueError("Mismatch between features and importances")

            df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Importance", y="Feature", data=df.head(20), ax=ax)
            st.pyplot(fig)
        else:
            st.info("⚠️ Feature importance not available for this model.")
    except Exception as e:
        st.error(f"❌ Feature importance plot failed: {e}")



# ------------------------------
# 🧪 Permutation Importance
# ------------------------------
def plot_permutation_importance(model_tuple, X_processed, module_name="loan"):
    st.subheader("🎯 Permutation Feature Importance")

    # Handle (model, feature_columns) or just model
    if isinstance(model_tuple, tuple):
        model, feature_columns = model_tuple
    else:
        model = model_tuple
        feature_columns = X_processed.columns.tolist()

    try:
        # Load dataset with labels
        data_path = f"data/{module_name}.csv"
        df = pd.read_csv(data_path)

        # Try to normalize label column to 'actual'
        if 'actual' not in df.columns:
            for col in df.columns:
                if col.lower() in ['class', 'label', 'target', 'fraud']:
                    df['actual'] = df[col]
                    break

        if 'actual' not in df.columns:
            st.warning("⚠️ Permutation importance requires a label column like 'Class'.")
            return

        # Ensure numeric columns and required features exist
        df = df.select_dtypes(include=[np.number]).fillna(0)

        if df['actual'].nunique() < 2:
            st.warning("⚠️ Permutation importance needs at least 2 classes (0 and 1) in 'actual'.")
            return

        missing = set(feature_columns) - set(df.columns)
        if missing:
            st.warning(f"⚠️ Missing features in dataset: {missing}")
            return

        X = df[feature_columns]
        y = df['actual']

        # Optional: scale features if needed
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

        # Run permutation importance
        result = permutation_importance(model, X_scaled, y, n_repeats=5, random_state=42)

        importances = result.importances_mean
        if importances.sum() == 0:
            st.warning("⚠️ All permutation importances are zero. The model may be uninformative.")
            return

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_idx = np.argsort(importances)
        ax.barh(np.array(feature_columns)[sorted_idx], importances[sorted_idx])
        ax.set_title("Permutation Importances")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Permutation importance failed: {e}")


# ------------------------------
# 🥧 Pie Chart
# ------------------------------
def plot_pie_chart(probability_score):
    st.subheader("🥧 Estimated Fraud Likelihood")

    if not isinstance(probability_score, (float, int)) or np.isnan(probability_score):
        probability_score = 0
    else:
        probability_score = max(0, min(1, probability_score))

    values = [probability_score, 1 - probability_score]
    labels = ['Fraud', 'Not Fraud']
    explode = [0.1 if v < 0.01 else 0 for v in values]

    def fmt(pct):
        return f"{pct:.5f}%" if pct < 0.01 else f"{pct:.1f}%"

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, explode=explode,
        autopct=lambda pct: fmt(pct), startangle=90,
        colors=['#ff6b6b', '#51cf66']
    )
    ax.axis('equal')
    st.pyplot(fig)

# ------------------------------
# 📋 Confusion Matrix + Report
# ------------------------------
def plot_confusion_report(y_true, y_pred):
    st.subheader("📋 Model Evaluation Report")
    try:
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
    except Exception as e:
        st.error(f"❌ Failed to generate confusion report: {e}")

# ------------------------------
# 📦 Box Plot
# ------------------------------
def plot_boxplot(df):
    st.subheader("📦 Feature Distribution")
    df = df.dropna(axis=1, how='all')

    if df.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.info("📭 No valid numeric features available for boxplot.")

# ------------------------------
# 🕸 Radar Chart
# ------------------------------
def plot_radar(model_scores):
    st.subheader("🕸 Radar Chart – Model Comparison")

    if not model_scores:
        st.warning("⚠️ No model scores available.")
        return

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
# 🌡️ Correlation Heatmap
# ------------------------------
def plot_correlation_heatmap(df, module):
    st.subheader("🌡️ Correlation Heatmap")
    module_df = load_dataset_for_module(module)

    df = module_df if module_df is not None else df
    input_features = df.loc[:, ~df.columns.str.endswith('_score')]

    if input_features.shape[1] > 1:
        corr = input_features.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, center=0,
            fmt=".2f", linewidths=0.5, linecolor='gray'
        )
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric features for correlation heatmap.")

# ------------------------------
# ⬇️ Download Report
# ------------------------------
def download_model_report(df, filename="fraud_report.csv"):
    st.download_button("⬇️ Download Model Report CSV", data=df.to_csv(index=False).encode(), file_name=filename, mime="text/csv")

# ------------------------------
# 🧾 Model Descriptions
# ------------------------------
def get_model_description(model_key):
    descriptions = {
        "rf": "🌲 **Random Forest**: Ensemble of decision trees.",
        "xgb": "🚀 **XGBoost**: Gradient boosting framework.",
        "lgbm": "🔆 **LightGBM**: Fast and scalable boosting.",
        "cat": "🐱 **CatBoost**: Handles categorical data well.",
        "lr": "📐 **Logistic Regression**: Simple and interpretable.",
        "iso": "🚨 **Isolation Forest**: Detects outliers in data."
    }
    return descriptions.get(model_key, "ℹ️ No description available.")
