# fraud_modules/credit_card.py

import pandas as pd
import numpy as np
import joblib
import traceback
import os

# -----------------------------
# 🔃 Load all credit card models
# -----------------------------
model_names = ["rf", "xgb", "cat", "lr", "iso"]
models = {}

print("🚀 Loading credit card models...")
for name in model_names:
    try:
        path = f"models/credit_card_{name}.pkl"
        if not os.path.exists(path):
            print(f"❌ Model file not found: {path}")
            continue

        pipe = joblib.load(path)
        feature_names = pipe.named_steps["pre"].get_feature_names_out() if "pre" in pipe.named_steps else None
        models[name] = (pipe, feature_names)
        print(f"✅ Loaded model: {name}")

    except Exception as e:
        print(f"❌ Failed loading model {name}: {e}")
        traceback.print_exc()

# -----------------------------
# 📦 Load fallback full dataset
# -----------------------------
try:
    full_data = pd.read_csv("data/creditcard.csv")
    print("✅ Loaded fallback dataset")
except Exception as e:
    print(f"❌ Could not load fallback dataset: {e}")
    full_data = None

# -----------------------------
# 🧠 Prediction Function
# -----------------------------
def predict_creditcard_fraud(df):
    print("🚦 Starting prediction...")
    
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or only NaNs.")

    df = df.copy()

    # 🎯 Extract actual labels if present
    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)

    df = df.select_dtypes(include=[np.number]).fillna(0)

    print(f"📊 Input shape after cleaning: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")

    scores = {}
    scored_df = df.copy()

    for name, (pipe, _) in models.items():
        try:
            print(f"🔍 Predicting with model: {name}")

            if name == "iso":
                preds = -pipe.decision_function(df)
                norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
                scored_df[f"{name}_score"] = norm
                scores[name] = norm.mean()
            else:
                preds = pipe.predict_proba(df)[:, 1]
                scored_df[f"{name}_score"] = preds
                scores[name] = preds.mean()

            print(f"✅ {name} score: {scores[name]:.4f}")

        except Exception as e:
            print(f"❌ Error in model {name}: {e}")
            traceback.print_exc()

    if not scores:
        raise ValueError("❌ No models could predict. (Check model load or input shape)")

    avg_score = np.mean(list(scores.values()))

    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values

    if len(df) < 5 and full_data is not None:
        print("🔁 Using fallback dataset for visualizations")
        fallback_df = full_data.select_dtypes(include=[np.number]).fillna(0)
        if "Class" in full_data.columns:
            fallback_df["actual"] = full_data["Class"]
        return avg_score, scores, fallback_df

    return avg_score, scores, scored_df

# -----------------------------
# 🌍 Globals for App Integration
# -----------------------------
models_plain = {k: v[0] for k, v in models.items()}
models_full = models
globals()["models"] = models_full
globals()["models_plain"] = models_plain
