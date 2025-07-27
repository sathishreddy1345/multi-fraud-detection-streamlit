import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# 🔃 Load all credit card models
# -----------------------------
model_names = ["rf", "xgb", "cat", "lr", "iso"]
models = {}

for name in model_names:
    try:
        pipe = joblib.load(f"credit_card_{name}.pkl")
        feature_names = pipe.named_steps["pre"].get_feature_names_out() if "pre" in pipe.named_steps else None
        models[name] = (pipe, feature_names)
        print(f"✅ Loaded model: {name}")
    except Exception as e:
        print(f"❌ Could not load model {name}: {e}")

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
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or only NaNs.")

    df = df.copy()

    if "Class" in df.columns:
        df["actual"] = df["Class"]
        df.drop(columns=["Class"], inplace=True)

    df = df.select_dtypes(include=[np.number]).fillna(0)

    scores = {}
    scored_df = df.copy()

    for name, (pipe, _) in models.items():
        try:
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

    if not scores:
        raise ValueError("❌ No models could predict.")

    avg_score = np.mean(list(scores.values()))

    if "actual" in df.columns:
        scored_df["actual"] = df["actual"].values

    if len(df) < 5 and full_data is not None:
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
