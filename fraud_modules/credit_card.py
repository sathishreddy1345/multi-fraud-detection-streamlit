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

# ✅ Load models properly (use native loader for xgb if needed)
for name in model_names:
    try:
        if name == 'xgb':
            model = XGBClassifier()
            model.load_model("models/credit_card_xgb.json")  # ✅ MUST use .json saved with .save_model()
            models[name] = model
        else:
            with open(f"models/credit_card_{name}.pkl", "rb") as f:
                models[name] = pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️ Model not found: credit_card_{name}")
    except Exception as e:
        print(f"❌ Error loading model {name}: {e}")

def predict_creditcard_fraud(df):
    df = df.copy()

    # ✅ Drop label/target column if exists
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    # ✅ Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # ✅ Check and fix feature mismatch (e.g., drop extra ID/column if needed)
    expected_features = 29
    if df.shape[1] > expected_features:
        df = df.iloc[:, :expected_features]

    df.fillna(0, inplace=True)
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

# ✅ Make models available to app.py
globals()['models'] = models
