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
model_features = {}

# ✅ Load models from the models/ directory
for name in model_names:
    try:
        path = f"models/credit_card_{name}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, tuple):
                models[name] = obj[0]
                model_features[name] = obj[1]
            else:
                models[name] = obj
    except FileNotFoundError:
        print(f"⚠️ Model not found: credit_card_{name}.pkl")
    except Exception as e:
        print(f"❌ Error loading model {name}: {e}")

def predict_creditcard_fraud(df):
    df = df.copy()

    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    scores = {}
    for name, model in models.items():
        try:
            features = model_features.get(name, df.columns[:model.n_features_in_])
            X = df[features].copy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if name == 'iso':
                score = (-model.decision_function(X_scaled)).mean()
            else:
                score = model.predict_proba(X_scaled)[:, 1].mean()

            scores[name] = score
            print(f"✅ {name} score: {score:.4f}")
        except Exception as e:
            print(f"❌ Error with model {name}: {e}")

    if not scores:
        raise ValueError("No valid models were able to make predictions.")

    final_score = np.mean(list(scores.values()))
    return final_score, scores, df

# ✅ Make models available globally
globals()['models'] = models
