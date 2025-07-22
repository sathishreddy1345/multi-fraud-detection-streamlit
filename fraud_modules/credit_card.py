# fraud_modules/credit_card.py

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --------------------
# 🔃 Load all models
# --------------------
model_names = ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']
models = {}
feature_columns = []

for name in model_names:
    try:
        with open(f"models/credit_card_{name}.pkl", "rb") as f:
            obj = pickle.load(f)
            model, features = obj if isinstance(obj, tuple) else (obj, None)
            models[name] = model
            if features:
                feature_columns = features
    except FileNotFoundError:
        print(f"⚠️ Model not found: credit_card_{name}")
    except Exception as e:
        print(f"❌ Error loading model {name}: {e}")

# --------------------
# 🧠 Predict Function
# --------------------
def predict_creditcard_fraud(df):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or contains only NaNs.")

    df = df.copy()

    # Drop label column if present
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    # Ensure only numeric columns
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    # Match expected features
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    else:
        raise ValueError("❌ Feature columns are not loaded from model pickle.")

    # 🧪 Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # 🎯 Predict with each model
    scores = {}
    for name, model in models.items():
        try:
            if name == 'iso':
                raw_scores = -model.decision_function(X_scaled)
                norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
                df[f'{name}_score'] = norm_scores
                scores[name] = norm_scores.mean()
            else:
                probs = model.predict_proba(X_scaled)[:, 1]
                df[f'{name}_score'] = probs
                scores[name] = probs.mean()

            print(f"✅ {name.upper()} model score: {scores[name]:.4f}")

        except Exception as e:
            print(f"❌ Error with model {name}: {e}")

    if not scores:
        raise ValueError("No valid models were able to make predictions.")

    # Final average score
    final_score = np.mean(list(scores.values()))
    return final_score, scores, df

# 🌍 Global export
globals()['models'] = models
