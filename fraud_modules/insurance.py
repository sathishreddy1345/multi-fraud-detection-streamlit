# fraud_modules/insurance.py

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# 🔃 Load all insurance models
# -----------------------------
model_names = ["rf", "xgb", "lgbm", "cat", "lr", "iso"]
models = {}

for name in model_names:
    try:
        with open(f"models/insurance_{name}.pkl", "rb") as f:
            obj = pickle.load(f)

            if isinstance(obj, tuple) and len(obj) == 2:
                model, features = obj
            else:
                model, features = obj, None

            models[name] = (model, features)

    except Exception as e:
        print(f"❌ Failed to load model insurance_{name}: {e}")

# -----------------------------
# 📦 Load fallback dataset
# -----------------------------
try:
    full_data = pd.read_csv("data/synthetic_insurance_fraud_refined.csv")
    print("✅ Loaded fallback synthetic insurance dataset")
except Exception as e:
    print(f"❌ Could not load fallback insurance dataset: {e}")
    full_data = None

# -----------------------------
# 🧠 Prediction Function
# -----------------------------
def predict_insurance_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or contains only NaNs.")

    df = df.copy()

    # ✅ Handle label column
    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)
    elif "fraud_reported" in df.columns:
        df["actual"] = df["fraud_reported"]
        df.drop(columns=["fraud_reported"], inplace=True)

    # ✅ Keep only numeric columns
    df = df.select_dtypes(include=[np.number]).fillna(0)

    scores = {}
    scored_df = df.copy()

    for key, (model, features) in models.items():
        try:
            if features:
                missing = set(features) - set(df.columns)
                if missing:
                    print(f"⚠️ Skipping model {key} - missing features: {missing}")
                    continue
                X_input = df[features]
            else:
                X_input = df

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_input)

            if key == "iso":
                raw = -model.decision_function(X_scaled)
                row_scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            else:
                row_scores = model.predict_proba(X_scaled)[:, 1]

            scores[key] = row_scores.mean()
            scored_df[f"{key}_score"] = row_scores
            print(f"✅ {key.upper()} score: {scores[key]:.4f}")

        except Exception as e:
            print(f"❌ Model {key} failed: {e}")
            import traceback
            traceback.print_exc()

    if not scores:
        raise ValueError("❌ No models succeeded in prediction.")

    final_score = np.mean(list(scores.values()))

    # ✅ Retain 'actual' if available
    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values

    # 🔁 Use fallback for visualization if needed
    if len(df) < 5 and full_data is not None:
        print("🔁 Using fallback dataset for visualizations")
        fallback_df = full_data.select_dtypes(include=[np.number]).fillna(0).copy()
        if "fraud_reported" in full_data.columns:
            fallback_df["actual"] = full_data["fraud_reported"].values
        return final_score, scores, fallback_df

    return final_score, scores, scored_df

# -----------------------------
# 🌐 Export model references
# -----------------------------
models_plain = {k: v[0] for k, v in models.items()}
models_full = models

globals()["models"] = models_plain
globals()["models_full"] = models_full
