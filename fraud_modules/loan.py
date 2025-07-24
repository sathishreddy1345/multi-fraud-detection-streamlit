# fraud_modules/loan.py

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

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
# 📦 Load full dataset (for visualization fallback)
# -----------------------------
try:
    full_data = pd.read_csv("data/loan_training_data.csv")
    print("✅ Loaded fallback loan training dataset")
except Exception as e:
    print(f"❌ Could not load fallback training data: {e}")
    full_data = None

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

    print("📊 Input columns:", df.columns.tolist())
    print("📊 Input shape:", df.shape)

    scores = {}
    scored_df = df.copy()

    for key, model in models.items():
        try:
            if features:
                missing = set(features) - set(df.columns)
                if missing:
                    raise ValueError(f"Missing features for model {key}: {missing}")
                X_input = df[features]
            else:
                X_input = df

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
            import traceback
            traceback.print_exc()

    if not scores:
        raise ValueError("❌ No models were able to predict.")

    final_score = np.mean(list(scores.values()))

    # -----------------------------
    # 🧪 Visualization Fallback Logic
    # -----------------------------

    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values

    if len(df) < 5 and full_data is not None:
        print("🔁 Using full dataset for visualizations due to small input size")
        return final_score, scores, full_data.select_dtypes(include=[np.number]).fillna(0)
        # Add actual labels if available
    
    return final_score, scores, scored_df


# -----------------------------
# 📦 Export cleaned model dict
# -----------------------------
# Use plain model dict for app-level visualizers
models_plain = {k: v[0] for k, v in models.items()}
models_full = models
globals()["models"] = models_plain         # For predictions
globals()["models_full"] = models_full     # For visualizations


# Keep full models with features for internal prediction
loan_models_with_features = models
