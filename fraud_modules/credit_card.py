# fraud_modules/credit_card.py
models = {}
for model_name in ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']:
    with open(f"models/credit_card_{model_name}.pkl", "rb") as f:
        models[model_name] = pickle.load(f)

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained models
models = {}
model_names = ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']

for model_name in model_names:
    try:
        with open(f"models/credit_card_{model_name}.pkl", "rb") as f:
            models[model_name] = pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️ Model {model_name} not found, skipping...")

def predict_creditcard_fraud(df):
    df = df.copy()

    # Drop target column if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

    # Keep only numeric features
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model_scores = {}

    for name, model in models.items():
        try:
            if name == 'iso':
                # Anomaly score: higher = more anomalous
                score = (-model.decision_function(X_scaled)).mean()
                model_scores[name] = score  # You can normalize later
            else:
                prob = model.predict_proba(X_scaled)[:, 1].mean()
                model_scores[name] = prob
        except Exception as e:
            print(f"⚠️ Model {name} prediction failed: {e}")

    if not model_scores:
        raise ValueError("❌ No predictions could be made — check model compatibility and input features.")

    final_score = np.mean(list(model_scores.values()))
    return final_score, model_scores, df
