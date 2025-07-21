# fraud_modules/credit_card.py

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

model_names = ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']
models = {}

# ✅ Load models from the models/ directory
scores = {}
for name, model in models.items():
    try:
        if name == 'iso':
            raw_scores = -model.decision_function(X_scaled)
            score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
            scores[name] = float(score.mean())
        else:
            probs = model.predict_proba(X_scaled)
            score = float(probs[:, 1].mean())
            scores[name] = score
        print(f"✅ {name} model score: {scores[name]:.4f}")
    except Exception as e:
        print(f"❌ Error with model {name}: {e}")


def predict_creditcard_fraud(df):
    df = df.copy()

    # Drop label column if exists
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    # Keep only numeric features
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    # ⚠️ Match expected feature count from model
    try:
        expected_features = next(iter(models.values())).n_features_in_
    except Exception:
        expected_features = 29

    # Adjust columns accordingly
    if df.shape[1] > expected_features:
        df = df.iloc[:, :expected_features]
    elif df.shape[1] < expected_features:
        raise ValueError(f"Input has {df.shape[1]} features, expected {expected_features}.")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    scores = {}

    for name, model in models.items():
        try:
            if name == 'iso':
                # IsolationForest anomaly score is reversed
                score = (-model.decision_function(X_scaled)).mean()
            else:
                score = model.predict_proba(X_scaled)[:, 1].mean()
            scores[name] = float(score)
            print(f"✅ {name} model score: {score:.4f}")
        except Exception as e:
            print(f"❌ Error with model {name}: {e}")

    if not scores:
        raise ValueError("No valid models were able to make predictions.")

    final_score = np.mean(list(scores.values()))
    return final_score, scores, df

# ✅ Expose models globally
globals()['models'] = models
