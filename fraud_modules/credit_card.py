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

# --------------------
# 🔃 Load all models
# --------------------
model_names = ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']
models = {}

for name in model_names:
    try:
        with open(f"models/credit_card_{name}.pkl", "rb") as f:
            obj = pickle.load(f)
            # Some models are saved as (model, features)
            model = obj[0] if isinstance(obj, tuple) else obj
            models[name] = model
    except FileNotFoundError:
        print(f"⚠️ Model not found: credit_card_{name}.pkl")
    except Exception as e:
        print(f"❌ Error loading model {name}: {e}")

# --------------------
# 🧠 Predict Function
# --------------------
def predict_creditcard_fraud(df):
    df = df.copy()

    # Drop label column if present
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    # Match feature size with model
    try:
        reference_model = next(iter(models.values()))
        expected_features = reference_model.n_features_in_
    except Exception:
        expected_features = 29  # fallback

    if df.shape[1] > expected_features:
        df = df.iloc[:, :expected_features]
    elif df.shape[1] < expected_features:
        raise ValueError(f"Input has {df.shape[1]} features; expected {expected_features}.")

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Score collection
    scores = {}
    for name, model in models.items():
        try:
            if name == 'iso':
                # Normalize IsolationForest score between 0 and 1
                raw_scores = -model.decision_function(X_scaled)
                score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
                scores[name] = score.mean()
            else:
                score = model.predict_proba(X_scaled)[:, 1].mean()
                scores[name] = score
            print(f"✅ {name.upper()} model score: {scores[name]:.4f}")
        except Exception as e:
            print(f"❌ Error with model {name}: {e}")

    if not scores:
        raise ValueError("No valid models were able to make predictions.")

    final_score = np.mean(list(scores.values()))
    return final_score, scores, df

# Export for app.py
globals()['models'] = models
