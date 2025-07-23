# fraud_modules/loan.py

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 🔃 Load all loan models
# -----------------------------
model_names = ["rf", "xgb", "lgbm", "cat", "lr", "iso"]
models = {}

for name in model_names:
    try:
        with open(f"models/loan_{name}.pkl", "rb") as f:
            obj = pickle.load(f)
            model = obj[0] if isinstance(obj, tuple) else obj
            features = obj[1] if isinstance(obj, tuple) else None
            models[name] = (model, features)
    except Exception as e:
        print(f"❌ Failed loading loan_{name}: {e}")

# -----------------------------
# 🧠 Predict Function
# -----------------------------
def predict_loan_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or contains only NaNs.")

    df = df.copy()

    # Drop label if exists
    if "Class" in df.columns:
        df.drop(columns=["Class"], inplace=True)

    # Keep numeric only
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    scores = {}
    scored_df = df.copy()

    for key, (model, features) in models.items():
        try:
            # Align features
            X_input = df[features] if features else df

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_input)

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

    if not scores:
        raise ValueError("❌ No models were able to predict.")

    final_score = np.mean(list(scores.values()))
    return final_score, scores, scored_df

# -----------------------------
# 📦 Export cleaned model dict
# -----------------------------
globals()["models"] = {k: v[0] for k, v in models.items()}
