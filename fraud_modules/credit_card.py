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

    THRESHOLD = 0.25   # ⚡ operating threshold (tuned for recall)

    for name, (pipe, _) in models.items():
        try:
            print(f"🔍 Predicting with model: {name}")

            if name == "iso":
                preds = -pipe.decision_function(df)
                norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
                proba = norm
            else:
                proba = pipe.predict_proba(df)[:, 1]

            # 👉 Keep true probability (model output)
            scored_df[f"{name}_prob"] = proba.round(4)

            # 👉 Human-friendly Risk Score (0–100) — NOT fake
            risk_score = (proba * 100).clip(0, 100)
            scored_df[f"{name}_risk_score"] = risk_score.round(2)

            # 👉 Risk Levels for UI / Charts
            scored_df[f"{name}_risk_level"] = pd.cut(
                risk_score,
                bins=[0, 20, 50, 100],
                labels=["Low", "Medium", "High"]
            )

            # 👉 Model decision using tuned threshold
            scored_df[f"{name}_flag"] = (proba >= THRESHOLD).astype(int)

            # 👉 Model-level summary score
            scores[name] = proba.mean()

            print(f"✅ {name} avg prob={scores[name]:.4f}")

        except Exception as e:
            print(f"❌ Error in model {name}: {e}")
            traceback.print_exc()

    if not scores:
        raise ValueError("❌ No models could predict. (Check model load or input shape)")

    # -------------------------
    # 🧮 Ensemble Aggregation
    # -------------------------
    scored_df["ensemble_prob"] = scored_df[[f"{m}_prob" for m in scores]].mean(axis=1)
    scored_df["ensemble_risk_score"] = (scored_df["ensemble_prob"] * 100).round(2)
    scored_df["ensemble_flag"] = (scored_df["ensemble_prob"] >= THRESHOLD).astype(int)

    scored_df["ensemble_risk_level"] = pd.cut(
        scored_df["ensemble_risk_score"],
        bins=[0, 20, 50, 100],
        labels=["Low", "Medium", "High"]
    )

    avg_score = scored_df["ensemble_prob"].mean()


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




def get_template_df():
    """
    Returns a zero-filled template dataframe
    using the same columns as the credit-card dataset.
    """

    cols = [
        'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
        'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
        'V21','V22','V23','V24','V25','V26','V27','V28',
        'Amount','Time'
    ]

    return pd.DataFrame([[0 for _ in cols]], columns=cols)

