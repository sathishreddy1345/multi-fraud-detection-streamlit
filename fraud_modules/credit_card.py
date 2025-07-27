# fraud_modules/credit_card.py

import pandas as pd
import numpy as np
import joblib
import os
import traceback

# -----------------------------
# 🔃 Load all credit card models
# -----------------------------
model_names = ["rf", "xgb", "cat", "lr", "iso"]
models = {}
model_errors = {}

print("🔁 Loading credit card models...")

for name in model_names:
    try:
        pipe = joblib.load(f"credit_card_{name}.pkl")
        feature_names = pipe.named_steps["pre"].get_feature_names_out() if "pre" in pipe.named_steps else None
        models[name] = (pipe, feature_names)
        print(f"✅ Loaded model: {name}")
    except Exception as e:
        print(f"❌ Could not load model {name}: {e}")
        model_errors[name] = str(e)

if not models:
    print("❗ No models loaded at all!")
else:
    print(f"📦 Total models loaded: {len(models)}")

# -----------------------------
# 📦 Load full fallback dataset
# -----------------------------
try:
    full_data = pd.read_csv("data/creditcard_generated_30k.csv")
    print("✅ Fallback dataset loaded")
except Exception as e:
    print(f"❌ Failed to load fallback dataset: {e}")
    full_data = None

# -----------------------------
# 🧠 Prediction Function
# -----------------------------
def predict_creditcard_fraud(df):
    print("🚀 Starting prediction pipeline...")

    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or all NaNs.")

    print("📊 Input shape:", df.shape)
    print("📊 Input columns:", df.columns.tolist())

    df = df.copy()

    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)
        print("🔁 Converted 'Class' to 'actual'")

    df = df.select_dtypes(include=[np.number]).fillna(0)
    print("🔢 Cleaned input shape:", df.shape)

    scores = {}
    scored_df = df.copy()
    prediction_errors = {}

    for name, (pipe, _) in models.items():
        print(f"\n🧪 Predicting with: {name}")
        try:
            if df.shape[0] == 0:
                raise ValueError("Input has 0 rows after cleaning.")

            if name == "iso":
                raw = -pipe.decision_function(df)
                norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                scores[name] = norm.mean()
                scored_df[f"{name}_score"] = norm
                print(f"✅ ISO mean score: {scores[name]:.4f}")
            else:
                proba = pipe.predict_proba(df)[:, 1]
                scores[name] = proba.mean()
                scored_df[f"{name}_score"] = proba
                print(f"✅ {name.upper()} mean prob: {scores[name]:.4f}")

        except Exception as e:
            print(f"❌ Error with model {name}: {e}")
            traceback.print_exc()
            prediction_errors[name] = str(e)

    if not scores:
        print("❌ No models succeeded. Details:")
        for name, err in prediction_errors.items():
            print(f"   - {name}: {err}")
        raise ValueError("❌ No models could predict.")

    avg_score = np.mean(list(scores.values()))
    print(f"\n🎯 Final average fraud score: {avg_score:.4f}")

    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values
        print("📌 Attached true labels to output")

    if len(df) < 5 and full_data is not None:
        print("🔁 Using fallback dataset for visualization due to small input size")
        fallback_df = full_data.select_dtypes(include=[np.number]).fillna(0)
        if "Class" in full_data.columns:
            fallback_df["actual"] = full_data["Class"]
        return avg_score, scores, fallback_df

    return avg_score, scores, scored_df

# -----------------------------
# 🌍 Global export for app
# -----------------------------
models_plain = {k: v[0] for k, v in models.items()}
models_full = models
globals()["models"] = models_full
globals()["models_plain"] = models_plain
