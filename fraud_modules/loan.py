# fraud_modules/loan.py

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_names = ["rf", "xgb", "lgbm", "cat", "lr", "iso"]
models = {}

# 🔃 Load all models
for name in model_names:
    try:
        with open(f"models/loan_{name}.pkl", "rb") as f:
            obj = pickle.load(f)
            model = obj[0] if isinstance(obj, tuple) else obj
            feature_columns = obj[1] if isinstance(obj, tuple) else None
            models[name] = (model, feature_columns)
    except Exception as e:
        print(f"❌ Failed loading {name}: {e}")

# 🔍 Prediction Function
def predict_loan_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or contains only NaNs.")

    df = df.copy()

    # Drop label column if present
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    # Keep numeric data only
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

scores = {}
scored_df = df.copy()
for key in models:
    model_info = models[key]
    if isinstance(model_info, tuple):
        model, features = model_info
    else:
        model = model_info
        features = None

    try:
        # Align features if available
        if features is not None:
            X_input = df[features]
        else:
            X_input = df

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

# Export model dictionary
globals()["models"] = {k: v[0] for k, v in models.items()}
