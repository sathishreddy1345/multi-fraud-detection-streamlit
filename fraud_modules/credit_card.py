# fraud_modules/credit_card.py

import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# 🔃 Load all credit card models
# -----------------------------
model_names = ["rf", "xgb", "cat", "lr", "iso"]
models = {}

print("🔍 Loading credit card models...")

for name in model_names:
    try:
        pipe = joblib.load(f"credit_card_{name}.pkl")
        feature_names = pipe.named_steps["pre"].get_feature_names_out() if "pre" in pipe.named_steps else None
        models[name] = (pipe, feature_names)
        print(f"✅ Loaded model: {name}, Features: {len(feature_names) if feature_names is not None else 'None'}")
    except Exception as e:
        print(f"❌ Could not load model {name}: {e}")

if not models:
    print("❗ No models were loaded. Make sure credit_card_*.pkl files exist.")

# -----------------------------
# 📦 Load full dataset (fallback)
# -----------------------------
try:
    full_data = pd.read_csv("data/creditcard_generated_30k.csv")
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

    print("🔍 Initial input shape:", df.shape)
    print("🔍 Initial input columns:", df.columns.tolist())

    df = df.copy()

    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)
        print("🔁 Mapped 'Class' to 'actual'")

    df = df.select_dtypes(include=[np.number]).fillna(0)
    print("🔢 Cleaned numeric input shape:", df.shape)

    scores = {}
    scored_df = df.copy()

    for name, (pipe, features) in models.items():
        try:
            print(f"\n🔍 Predicting with model: {name.upper()}")
            X_input = df.copy()

            if X_input.shape[0] == 0:
                print(f"⚠️ Skipping {name} due to empty input")
                continue

            if name == "iso":
                raw = -pipe.decision_function(X_input)
                norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                scores[name] = norm.mean()
                scored_df[f"{name}_score"] = norm
                print(f"✅ ISO normalized score mean: {scores[name]:.4f}")
            else:
                if hasattr(pipe, "predict_proba"):
                    prob = pipe.predict_proba(X_input)[:, 1]
                    scores[name] = prob.mean()
                    scored_df[f"{name}_score"] = prob
                    print(f"✅ {name.upper()} prob mean: {scores[name]:.4f}")
                else:
                    print(f"⚠️ Model {name} does not support predict_proba")

        except Exception as e:
            print(f"❌ Error in model {name}: {e}")
            import traceback
            traceback.print_exc()

    if not scores:
        raise ValueError("❌ No models could predict.")

    avg_score = np.mean(list(scores.values()))
    print(f"\n🎯 Final average fraud score: {avg_score:.4f}")

    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values
        print("✅ Attached true labels to scored_df")

    if len(df) < 5 and full_data is not None:
        print("📉 Input too small, returning fallback dataset for visualization")
        fallback_df = full_data.select_dtypes(include=[np.number]).fillna(0)
        if "Class" in full_data.columns:
            fallback_df["actual"] = full_data["Class"]
        return avg_score, scores, fallback_df

    return avg_score, scores, scored_df

# -----------------------------
# 🌍 Global for App Use
# -----------------------------
models_plain = {k: v[0] for k, v in models.items()}
models_full = models
globals()["models"] = models_full
globals()["models_plain"] = models_plain
