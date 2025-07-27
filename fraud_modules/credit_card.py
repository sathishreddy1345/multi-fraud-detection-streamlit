# fraud_modules/credit_card.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# 🔃 Load all credit card models
# -----------------------------
model_names = ["lr", "rf", "xgb", "cat", "iso"]
models = {}
models_full = {}

for name in model_names:
    try:
        path = f"models/credit_card_{name}.pkl"
        model = joblib.load(path)
        feature_names = model.named_steps["pre"].get_feature_names_out()
        models[name] = model
        models_full[name] = (model, feature_names)
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")

# -----------------------------
# 📦 Load fallback dataset
# -----------------------------
try:
    full_data = pd.read_csv("data/creditcard_generated_30k.csv")
    full_data.rename(columns={"Class": "actual"}, inplace=True)
    print("✅ Fallback dataset loaded for visualization")
except Exception as e:
    print(f"⚠️ Could not load fallback dataset: {e}")
    full_data = None

# -----------------------------
# 🧠 Prediction Function
# -----------------------------
def predict_creditcard_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or invalid.")

    df = df.copy()

    # Rename label if present
    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)

    df = df.select_dtypes(include=[np.number]).fillna(0)

    print("📊 Input shape:", df.shape)
    scores = {}
    result_df = df.copy()

    for name, (model, features) in models_full.items():
        try:
            # Ensure feature alignment
            for col in features:
                if col not in df.columns:
                    df[col] = 0
            X = df[features]

            # Predict
            if name == "iso":
                raw = -model.decision_function(X)
                norm_scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                scores[name] = norm_scores.mean()
                result_df[f"{name}_score"] = norm_scores
            else:
                prob = model.predict_proba(X)[:, 1]
                scores[name] = prob.mean()
                result_df[f"{name}_score"] = prob

            print(f"✅ {name.upper()} score: {scores[name]:.4f}")

        except Exception as e:
            print(f"❌ Failed model {name}: {e}")

    if not scores:
        raise ValueError("❌ No models could predict.")

    final_score = np.mean(list(scores.values()))

    # Attach actual if present
    if "actual" in df.columns:
        result_df["actual"] = df["actual"].values

    # 🔁 Fallback for visualizer
    if len(df) < 5 and full_data is not None:
        fallback_df = full_data.select_dtypes(include=[np.number]).copy()
        if "actual" in full_data.columns:
            fallback_df["actual"] = full_data["actual"]
        return final_score, scores, fallback_df

    return final_score, scores, result_df

# -----------------------------
# 🌐 Export
# -----------------------------
globals()["models"] = models
globals()["models_full"] = models_full
