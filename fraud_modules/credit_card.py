# credit_card.py
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ✅ Load Models
model_names = ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']
models = {}

for name in model_names:
    try:
        with open(f"models/credit_card_{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    except:
        print(f"⚠️ Could not load: {name}")

def predict_creditcard_fraud(df):
    df = df.copy()

    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    df = df.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model_scores = {}
    for name, model in models.items():
        try:
            if name == 'iso':
                score = -model.decision_function(X_scaled).mean()
                model_scores[name] = score
            else:
                prob = model.predict_proba(X_scaled)[:, 1].mean()
                model_scores[name] = prob
        except Exception as e:
            print(f"❌ {name} prediction failed: {e}")

    final_score = np.mean(list(model_scores.values()))
    return final_score, model_scores, df

# expose models for SHAP
globals()['models'] = models
