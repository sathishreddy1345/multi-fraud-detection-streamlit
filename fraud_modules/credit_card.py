import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained models safely
models = {}
for model_name in ['rf', 'xgb', 'lgbm', 'cat', 'lr']:
    model_path = f"models/credit_card_{model_name}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)
    else:
        print(f"⚠️ Model not found: {model_path}")

def predict_creditcard_fraud(df):
    if not models:
        raise ValueError("❌ No models loaded for Credit Card fraud detection.")

    df = df.copy()

    # Drop target column if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Handle missing values
    df = df.fillna(0)

    # Scale input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Predict using available models
    model_scores = {}
    for name, model in models.items():
        try:
            prob = model.predict_proba(X_scaled)[:, 1].mean()
            model_scores[name] = prob
        except Exception as e:
            print(f"Error predicting with model {name}: {e}")

    if not model_scores:
        raise ValueError("❌ No predictions could be made — check model compatibility and input features.")

    # Final average score
    final_score = np.mean(list(model_scores.values()))
    return final_score, model_scores, df
