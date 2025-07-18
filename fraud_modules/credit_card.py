# fraud_modules/credit_card.py

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# 📦 Load pre-trained models
models = {}
model_names = ["rf", "xgb", "lgbm", "cat", "lr", "iso"]
for name in model_names:
    try:
        with open(f"models/creditcard_{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    except Exception as e:
        print(f"⚠️ Could not load model {name}: {e}")

# 🧠 Predict Credit Card Fraud
def predict_creditcard_fraud(df: pd.DataFrame):
    # 💡 Preprocessing
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])  # drop label if present
    X = df.copy()
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = {}
    for name, model in models.items():
        if name == "iso":
            pred = -model.decision_function(X_scaled)
            score = pred.mean()
        else:
            pred = model.predict_proba(X_scaled)[:, 1]
            score = pred.mean()
        scores[name] = round(score, 4)

    # 🔢 Weighted score (you can customize this logic)
    combined_score = sum(scores.values()) / len(scores)

    return combined_score, scores, X  # X is processed input for SHAP
