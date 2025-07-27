# fraud_modules/credit_card.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 🔃 Load all credit card models
# -----------------------------
model_names = ["rf", "xgb", "cat", "lr", "iso"]
models = {}

for name in model_names:
    try:
        obj = joblib.load(f"creditcard_model_{name}.pkl")
        model = obj
        features = None
        if hasattr(model.named_steps["pre"], "get_feature_names_out"):
            features = model.named_steps["pre"].get_feature_names_out()
        models[name] = (model, features)
        print(f"✅ Loaded creditcard_model_{name}.pkl")
    except Exception as e:
        print(f"❌ Failed to load model {name}: {e}")

# -----------------------------
# 📦 Load full dataset (for fallback visualization)
# -----------------------------
try:
    full_data = pd.read_csv("data/creditcard_generated_30k.csv")
    print("✅ Loaded full fallback credit card dataset")
except Exception as e:
    print(f"❌ Failed to load full dataset: {e}")
    full_data = None

# -----------------------------
# 🧠 Prediction Function
# -----------------------------
def predict_creditcard_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or only NaNs.")

    df = df.copy()

    # 🏷️ Rename label if needed
    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)

    df = df.select_dtypes(include=[np.number]).fillna(0)

    scores = {}
    scored_df = df.copy()

    for name, (pipe, features) in models.items():
        try:
            X_input = df.copy()

            # Fit scaler and transform
            if features is not None:
                missing = set(features) - set(pipe.named_steps["pre"].get_feature_names_out())
                if missing:
                    raise ValueError(f"Missing features for {name}: {missing}")

            if name == "iso":
                raw = -pipe.decision_function(X_input)
                norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                scores[name] = norm.mean()
                scored_df[f"{name}_score"] = norm
            else:
                probs = pipe.predict_proba(X_input)[:, 1]
                scores[name] = probs.mean()
                scored_df[f"{name}_score"] = probs

            print(f"✅ {name.upper()} model score: {scores[name]:.4f}")

        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    if not scores:
        raise ValueError("❌ No models could predict.")

    final_score = np.mean(list(scores.values()))

    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values

    if len(df) < 5 and full_data is not None:
        fallback_df = full_data.select_dtypes(include=[np.number]).copy()
        if "Class" in full_data.columns:
            fallback_df["actual"] = full_data["Class"]
        return final_score, scores, fallback_df

    return final_score, scores, scored_df

# -----------------------------
# 🌐 Global Export for App
# -----------------------------
models_plain = {k: v[0] for k, v in models.items()}
models_full = models

globals()["models"] = models_full
globals()["models_plain"] = models_plain
