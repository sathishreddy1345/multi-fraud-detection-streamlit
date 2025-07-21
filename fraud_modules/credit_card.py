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
for name in model_names:
    try:
        path = f"models/credit_card_{name}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
            # In case model was saved as (model, feature_columns)
            models[name] = obj[0] if isinstance(obj, tuple) else obj
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

    # ⚠️ Feature fix: match trained model's feature count
    try:
        expected_features = models['rf'].n_features_in_
        if df.shape[1] > expected_features:
            df = df.iloc[:, :expected_features]
        elif df.shape[1] < expected_features:
            raise ValueError(f"Input has {df.shape[1]} features, expected {expected_features}.")
    except:
        expected_features = 29
        df = df.iloc[:, :expected_features]

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
        raise ValueError("No valid models were able to make predictions.")

    final_score = np.mean(list(scores.values()))
    return final_score, scores, df

# ✅ Make models available globally
globals()['models'] = models
