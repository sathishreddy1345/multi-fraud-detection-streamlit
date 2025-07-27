# fraud_modules/credit_card.py

import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# 🔃 Load all credit card models
# -----------------------------
model_names = ["rf", "xgb", "cat", "lr", "iso"]
models = {}

for name in model_names:
    try:
        # Load full pipeline
        pipe = joblib.load(f"credit_card_{name}.pkl")
        # Try to get feature names from preprocessor (optional)
        feature_names = None
        if "pre" in pipe.named_steps:
            preprocessor = pipe.named_steps["pre"]
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
        models[name] = (pipe, feature_names)
        print(f"✅ Loaded model: {name}")
    except Exception as e:
        print(f"❌ Could not load model {name}: {e}")

# -----------------------------
# 📦 Load fallback dataset
# -----------------------------
try:
    full_data = pd.read_csv("data/creditcard_generated_30k.csv")
    print("✅ Loaded fallback dataset for credit card fraud")
except Exception as e:
    print(f"❌ Could not load fallback dataset: {e}")
    full_data = None

# -----------------------------
# 🧠 Prediction Function
# -----------------------------
def predict_creditcard_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or contains only NaNs.")

    df = df.copy()

    # Handle 'Class' column
    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)

    # Only use numeric columns
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("❌ After preprocessing, no usable data remains.")

    scores = {}
    scored_df = df.copy()

    for name, (pipe, features) in models.items():
        try:
            X_input = df.copy()
            # Fit pipeline (should already include preprocessor)
            if name == "iso":
                raw = -pipe.decision_function(X_input)
                norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                scores[name] = norm.mean()
                scored_df[f"{name}_score"] = norm
            else:
                prob = pipe.predict_proba(X_input)[:, 1]
                scores[name] = prob.mean()
                scored_df[f"{name}_score"] = prob

            print(f"✅ {name.upper()} model score: {scores[name]:.4f}")

        except Exception as e:
            print(f"❌ Error in model {name}: {e}")

    if not scores:
        raise ValueError("❌ No models could predict.")

    # Final average score
    final_score = np.mean(list(scores.values()))

    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values

    # If too few rows, use fallback for visualizer
    if len(df) < 5 and full_data is not None:
        print("🔁 Using fallback dataset for visualizations.")
        fallback_df = full_data.select_dtypes(include=[np.number]).fillna(0)
        if "Class" in full_data.columns:
            fallback_df["actual"] = full_data["Class"]
        return final_score, scores, fallback_df

    return final_score, scores, scored_df

# -----------------------------
# 🌐 Global Model Access
# -----------------------------
models_plain = {k: v[0] for k, v in models.items()}
models_full = models

globals()["models"] = models_full
globals()["models_plain"] = models_plain
