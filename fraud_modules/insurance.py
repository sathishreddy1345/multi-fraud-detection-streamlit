# fraud_modules/insurance.py

import pickle
import pandas as pd
import numpy as np
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
                model = obj
                features = None
            models[name] = {"model": model, "features": features}
    except Exception as e:
        print(f"❌ Failed loading insurance_{name}.pkl: {e}")

# -----------------------------
# 📦 Load full fallback dataset
# -----------------------------
try:
    full_data = pd.read_csv("data/insurance.csv")
    print("✅ Loaded fallback insurance dataset")
except Exception as e:
    print(f"❌ Could not load fallback insurance data: {e}")
    full_data = None

# -----------------------------
# 🧠 Predict Function
# -----------------------------
def predict_insurance_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or invalid.")

    df = df.copy()

    # Retain labels for permutation
    if "fraud_reported" in df.columns:
        df["actual"] = df["fraud_reported"]
        df.drop(columns=["fraud_reported"], inplace=True)

    df = df.select_dtypes(include=[np.number]).fillna(0)

    print("📊 Input columns:", df.columns.tolist())
    print("📊 Input shape:", df.shape)

    scores = {}
    scored_df = df.copy()

    for key, entry in models.items():
        try:
            model = entry["model"]
            features = entry.get("features")

            if features is not None:
                if not isinstance(features, list):
                    raise TypeError(f"Invalid feature list for model {key}")
                missing = set(features) - set(df.columns)
                if missing:
                    print(f"⚠️ Skipping model {key} - missing features: {missing}")
                    continue
                X = df[features]
            else:
                X = df

            # Scale numeric data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if key == "iso":
                raw = -model.decision_function(X_scaled)
                row_scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            else:
                row_scores = model.predict_proba(X_scaled)[:, 1]

            scores[key] = float(np.mean(row_scores))
            scored_df[f"{key}_score"] = row_scores

            print(f"✅ {key.upper()} model score: {scores[key]:.4f}")

        except Exception as e:
            print(f"❌ Model {key} failed: {e}")

    if not scores:
        raise ValueError("❌ No models succeeded in prediction.")

    # Use fallback data for visualization if too few rows
    if len(df) < 5 and full_data is not None:
        print("🔁 Using full dataset for visualizations due to small input size")
        fallback_df = full_data.select_dtypes(include=[np.number]).fillna(0).copy()
        if "fraud_reported" in full_data.columns:
            fallback_df["actual"] = full_data["fraud_reported"]
        return np.mean(list(scores.values())), scores, fallback_df

    # Ensure actual label exists in output
    if "actual" in df.columns:
        scored_df["actual"] = df["actual"]

    return np.mean(list(scores.values())), scores, scored_df

# Expose model dicts to app
models_plain = {k: v["model"] for k, v in models.items()}
models_full = models
globals()["models"] = models_plain
globals()["models_full"] = models_full
