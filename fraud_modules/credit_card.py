# fraud_modules/credit_card.py

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 🔃 Load all credit card models
# -----------------------------
model_names = ["rf", "xgb", "lgbm", "cat", "lr", "iso"]
models = {}

for name in model_names:
    try:
        with open(f"models/credit_card_{name}.pkl", "rb") as f:
            obj = pickle.load(f)
            model = obj[0] if isinstance(obj, tuple) else obj
            features = obj[1] if isinstance(obj, tuple) else None
            models[name] = (model, features)
    except Exception as e:
        print(f"❌ Failed loading credit_card_{name}: {e}")

# -----------------------------
# 📦 Load full dataset (for fallback visualizations)
# -----------------------------
try:
    full_data = pd.read_csv("data/creditcard_balanced.csv")
    print("✅ Loaded fallback credit card dataset for visualization")
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

    # 🏷️ Rename label if needed
    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)

    # 🔢 Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    print("📊 Input columns:", df.columns.tolist())
    print("📊 Input shape:", df.shape)

    scores = {}
    scored_df = df.copy()

    for name, (model, features) in models.items():
        try:
            if features:
                missing = set(features) - set(df.columns)
                if missing:
                    raise ValueError(f"Missing features for model {name}: {missing}")
                X = df[features]
            else:
                X = df

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if name == "iso":
                raw_scores = -model.decision_function(X_scaled)
                norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
                scored_df[f"{name}_score"] = norm_scores
                scores[name] = norm_scores.mean()
            else:
                probs = model.predict_proba(X_scaled)[:, 1]
                scored_df[f"{name}_score"] = probs
                scores[name] = probs.mean()

            print(f"✅ {name.upper()} model score: {scores[name]:.4f}")

        except Exception as e:
            print(f"❌ Error in model {name}: {e}")
            import traceback
            traceback.print_exc()

    if not scores:
        raise ValueError("❌ No models were able to predict.")

    final_score = np.mean(list(scores.values()))

    # Attach true labels if available
    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values

    # Visual fallback
    if len(df) < 5 and full_data is not None:
        print("🔁 Using full dataset for visualizations due to small input size")
        fallback = full_data.select_dtypes(include=[np.number]).fillna(0).copy()

        if "Class" in full_data.columns:
            fallback["actual"] = full_data["Class"]

        return final_score, scores, fallback

    return final_score, scores, scored_df

# -----------------------------
# 🌐 Expose models
# -----------------------------
models_plain = {k: v[0] for k, v in models.items()}
models_full = models

globals()["models"] = models_full
globals()["models_plain"] = models_plain
