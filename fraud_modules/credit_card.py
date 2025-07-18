import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained models
models = {}
for model_name in ['rf', 'xgb', 'lgbm', 'cat', 'lr']:
    with open(f"models/credit_card_{model_name}.pkl", "rb") as f:
        models[model_name] = pickle.load(f)

def predict_creditcard_fraud(df):
    df = df.copy()

    # Drop target column if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

    # Drop non-numeric columns if any (example: transaction_id)
    df = df.select_dtypes(include=[np.number])

    # Fill or drop missing values
    df = df.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Predict with all models
    model_scores = {}
    for name, model in models.items():
        prob = model.predict_proba(X_scaled)[:, 1].mean()
        model_scores[name] = prob

    # Combine score (average)
    final_score = np.mean(list(model_scores.values()))

    return final_score, model_scores, df
