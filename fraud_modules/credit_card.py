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

# ✅ Load pre-trained models
for name in model_names:
    try:
        if name == 'xgb':
            model = XGBClassifier()
            model.load_model("models/credit_card_xgb.json")  # Only JSON works for XGBoost 2.x
            models[name] = model
        else:
            with open(f"models/credit_card_{name}.pkl", "rb") as f:  # ✅ Fixed from loan_ to credit_card_
                obj = pickle.load(f)
                if isinstance(obj, tuple):
                    models[name] = obj[0]  # Extract model from (model, features)
                else:
                    models[name] = obj

    except FileNotFoundError:
        print(f"⚠️ Model not found: credit_card_{name}")
    except Exception as e:
        print(f"❌ Error loading model {name}: {e}")

def predict_creditcard_fraud(df):
    df = df.copy()

    # ✅ Remove label if present
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    # ✅ Ensure only numerical input
    df = df.select_dtypes(include=[np.number])

    # ✅ Trim or validate features
    expected_features = 29
    if df.shape[1] > expected_features:
        df = df.iloc[:, :expected_features]
    elif df.shape[1] < expected_features:
        raise ValueError(f"Input has {df.shape[1]} features, expected {expected_features}.")

    # ✅ Standardize
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
        raise ValueError("No valid models were able to make predictions.")

    final_score = np.mean(list(scores.values()))
    return final_score, scores, df

# ✅ Make models available globally
globals()['models'] = models
