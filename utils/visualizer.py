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

    try:
        # Step 1: Unpack model
        if isinstance(model_tuple, tuple):
            model, _ = model_tuple
        else:
            model = model_tuple

        feature_names = None

        # Step 2: If pipeline, extract from preprocessor
        if hasattr(model, "named_steps"):
            steps = model.named_steps

            # Try to extract preprocessor feature names
            for name, step in steps.items():
                if hasattr(step, "get_feature_names_out"):
                    feature_names = step.get_feature_names_out()
                    break  # use first match

            # Get actual model inside pipeline
            for step in reversed(steps.values()):
                if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                    model = step
                    break

        # Step 3: Get importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            st.warning("⚠️ This model does not support feature importance.")
            return

        # Step 4: Fallback feature names
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        # Step 5: Plot
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=df.head(20), ax=ax)
        ax.set_title("Top Feature Importances")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Feature importance plot failed: {e}")

# ------------------------------
# 🧪 Permutation Importance
# ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

def plot_permutation_importance(model_tuple=None, module="loan"):
    st.subheader("🎯 Permutation Feature Importance (Dataset-Based)")

    # Map each module to its dataset
    dataset_paths = {
        "loan": "data/loan.csv",
        "insurance": "data/insurance.csv",
        "credit_card": "data/creditcard.csv",
        "paysim": "data/paysim.csv"
    }

    path = dataset_paths.get(module.lower())
    if not path:
        st.error(f"❌ No dataset found for module '{module}'")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return

    # Try to find a target column
    target_col = next((col for col in df.columns if col.lower() in ['class', 'label', 'fraud', 'fraud_reported', 'target', 'actual']), None)
    if not target_col:
        st.error("❌ No valid target column found.")
        return

    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode string labels
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    # Fit a simple model
    try:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        result = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=42)
        importances = result.importances_mean
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(X.columns[sorted_idx], importances[sorted_idx])
        ax.set_title("Permutation Importances")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Permutation importance failed: {e}")


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

def detect_module_from_df(df):
    module_features = {
        "loan": {"income", "loan_amount", "credit_score", "employment_years"},
        "insurance": {"policy_annual_premium", "umbrella_limit", "auto_make"},
        "credit_card": {"V1", "V2", "V3", "Amount", "Time"},
        "paysim": {"type", "oldbalanceOrg", "newbalanceOrig", "isFraud"},
    }

    df_cols = set(df.columns.str.lower())
    best_match = None
    max_overlap = 0

    for module, features in module_features.items():
        overlap = len(df_cols & set(f.lower() for f in features))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = module

    return best_match or "loan"  # fallback

def plot_correlation_heatmap(df, module=None):
    st.subheader("🌡️ Correlation Heatmap")

    if module is None:
        module = detect_module_from_df(df)

    # Fallback if df is invalid
    if df is None or df.empty or df.isnull().all().all():
        module_df = load_dataset_for_module(module)
        df = module_df if module_df is not None else df

    input_features = df.loc[:, ~df.columns.str.endswith('_score')]
    input_features = input_features.select_dtypes(include=[np.number])

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
        st.warning("⚠️ Not enough numeric features for correlation heatmap.")


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
