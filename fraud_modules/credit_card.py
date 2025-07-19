# fraud_modules/credit_card.py

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ✅ Load all 6 models if available
model_names = ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']
models = {}

for name in model_names:
    try:
        with open(f"models/credit_card_{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️ Model '{name}' not found in models/ folder — skipping.")

def predict_creditcard_fraud(df):
    df = df.copy()

    # ✅ Drop target if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

    # ✅ Use only numeric input features
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(0)

    # ✅ Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model_scores = {}

    # ✅ Loop over all models
    for name, model in models.items():
        try:
            if name == 'iso':
                score = (-model.decision_function(X_scaled)).mean()  # Higher score = more anomalous
                model_scores[name] = score
            else:
                prob = model.predict_proba(X_scaled)[:, 1].mean()
                model_scores[name] = prob
        except Exception as e:
            print(f"⚠️ Model '{name}' prediction failed: {e}")

    if not model_scores:
        raise ValueError("❌ No predictions could be made — check model files and input features.")

    # ✅ Average fraud score across all models
    final_score = np.mean(list(model_scores.values()))

    return final_score, model_scores, df
