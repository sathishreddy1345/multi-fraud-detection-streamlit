# fraud_modules/credit_card.py

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

model_names = ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']
models = {}

# ✅ Load models from /models folder
for name in model_names:
    try:
        with open(f"models/credit_card_{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️ Model not found: credit_card_{name}.pkl")

def predict_creditcard_fraud(df):
    df = df.copy()

    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    df = df.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    scores = {}

    for name, model in models.items():
        try:
            if name == 'iso':
                score = (-model.decision_function(X_scaled)).mean()
            else:
                score = model.predict_proba(X_scaled)[:, 1].mean()
            scores[name] = score
        except Exception as e:
            print(f"❌ Error with model {name}: {e}")

    if not scores:
        raise ValueError("No valid models for prediction.")

    final_score = np.mean(list(scores.values()))
    return final_score, scores, df

# Allow app.py to access models
globals()['models'] = models
