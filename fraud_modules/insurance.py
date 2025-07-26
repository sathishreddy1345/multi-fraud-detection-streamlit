# fraud_modules/insurance.py

import pickle
import numpy as np
import pandas as pd
import os

# -----------------------------
# 🔃 Load all insurance models
# -----------------------------
model_names = ["rf", "xgb", "cat", "lr", "iso"]
models = {}

default_features = [
    'months_as_customer', 'age', 'policy_state', 'policy_deductible',
    'policy_annual_premium', 'umbrella_limit', 'auto_make',
    'auto_year', 'total_claim_amount', 'vehicle_claim'
]

for name in model_names:
    try:
        with open(f"models/insurance_{name}.pkl", "rb") as f:
            obj = pickle.load(f)

            # Correct logic: check if model was saved as (model, features) or just model
            if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], list):
                model, features = obj
            else:
                model = obj
                features = default_features

            models[name] = (model, features)

    except Exception as e:
        print(f"❌ Failed loading insurance_{name}: {e}")

# -----------------------------
# 📦 Load fallback dataset
# -----------------------------
try:
    full_data = pd.read_csv("data/insurance.csv")
    print("✅ Loaded fallback insurance training dataset")
except Exception as e:
    print(f"❌ Could not load fallback insurance dataset: {e}")
    full_data = None

# -----------------------------
# 🧠 Predict Function
# -----------------------------
def predict_insurance_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or contains only NaNs.")

    df = df.copy()

    # Handle label column
    if "fraud_reported" in df.columns:
        df["actual"] = df["fraud_reported"]
        df.drop(columns=["fraud_reported"], inplace=True)

    df.fillna(0, inplace=True)  # Keep all features as-is

    print("📊 Input columns:", df.columns.tolist())
    for key, (_, features) in models.items():
        print(f"🔍 Checking model '{key}' required features...")
        missing = set(features) - set(df.columns)
        if missing:
            print(f"❌ Model {key} missing columns: {missing}")
        else:
            print(f"✅ Model {key} received all required columns.")


    print("📊 Input shape:", df.shape)

    scores = {}
    scored_df = df.copy()

    for key, (model, features) in models.items():
        try:
            if features:
                missing = set(features) - set(df.columns)
                if missing:
                    raise ValueError(f"Missing features for model {key}: {missing}")
                X_input = df[features]
            else:
                X_input = df

            # No manual scaling here — model pipeline handles everything
            X_scaled = X_input

            if key == "iso":
                raw = -model.decision_function(X_scaled)
                row_scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                scores[key] = row_scores.mean()
                scored_df[f"{key}_score"] = row_scores
            else:
                row_scores = model.predict_proba(X_scaled)[:, 1]
                scores[key] = row_scores.mean()
                scored_df[f"{key}_score"] = row_scores

            print(f"✅ {key.upper()} model score: {scores[key]:.4f}")

        except Exception as e:
            print(f"❌ Error in model {key}: {e}")
            import traceback
            traceback.print_exc()

    if not scores:
        raise ValueError("❌ No models were able to predict.")

    final_score = np.mean(list(scores.values()))

    # -----------------------------
    # Visualization fallback logic
    # -----------------------------
    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values

    if len(df) < 5 and full_data is not None:
        print("🔁 Using full dataset for visualizations")
        fallback_df = full_data.select_dtypes(include=[np.number]).fillna(0).copy()
        if "fraud_reported" in full_data.columns:
            fallback_df["actual"] = full_data["fraud_reported"].values
        return final_score, scores, fallback_df

    return final_score, scores, scored_df

# -----------------------------
# 📦 Export cleaned model dict
# -----------------------------
models_plain = {k: v[0] for k, v in models.items()}
models_full = models
globals()["models"] = models_plain
globals()["models_full"] = models_full

# Retain for direct access
insurance_models_with_features = models
