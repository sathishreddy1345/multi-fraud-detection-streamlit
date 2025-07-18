import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_names = ["rf", "xgb", "lgbm", "cat", "lr", "iso"]
models = {}
for name in model_names:
    with open(f"models/insurance_{name}.pkl", "rb") as f:
        models[name] = pickle.load(f)

scaler = StandardScaler()

def predict_insurance_fraud(df):
    global models
    X = df.copy()
    X = X.select_dtypes(include="number").fillna(0)
    X_scaled = scaler.fit_transform(X)

    predictions = {}
    for key, model in models.items():
        if key == "iso":
            preds = model.predict(X_scaled)
            scores = np.where(preds == -1, 1, 0)
        else:
            scores = model.predict_proba(X_scaled)[:, 1]
        predictions[key] = np.mean(scores)

    avg_score = np.mean(list(predictions.values()))
    return avg_score, predictions, pd.DataFrame(X_scaled, columns=X.columns)
