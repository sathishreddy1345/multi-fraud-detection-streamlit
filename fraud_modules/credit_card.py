# fraud_modules/credit_card.py

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --------------------
# 🔃 Load all models
# --------------------
model_names = ['rf', 'xgb', 'lgbm', 'cat', 'lr', 'iso']
models = {}
models_full = {}
feature_columns = []

for name in model_names:
    try:
        with open(f"models/credit_card_{name}.pkl", "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, tuple):
                model, features = obj
                models[name] = model
                models_full[name] = obj
                if features:
                    feature_columns = features  # Last loaded wins; ensure consistency
            else:
                models[name] = obj
    except FileNotFoundError:
        print(f"⚠️ Model not found: credit_card_{name}.pkl")
    except Exception as e:
        print(f"❌ Error loading model {name}: {e}")

# --------------------
# 🧠 Predict Function
# --------------------
def predict_creditcard_fraud(df: pd.DataFrame):
    if df.empty or df.isnull().all().all():
        raise ValueError("Input dataframe is empty or contains only NaNs.")

    df = df.copy()

    # 🧹 Drop label column if present
    for col in ['Class', 'actual']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 🧪 Ensure numeric input only
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    # 🧱 Align columns with training
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    else:
        raise ValueError("❌ Feature columns were not loaded with model pickle files.")

    # ⚖️ Scale input
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # 🔍 Predict using all models
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
            print(f"✅ {name.upper()} score: {scores[name]:.4f}")
        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    if not scores:
        raise ValueError("❌ No valid model could produce predictions.")

    final_score = np.mean(list(scores.values()))
    df["actual"] = df.get("actual", None)

    return final_score, scores, df

# 🌍 Export for app use
globals()["models"] = models
globals()["models_full"] = models_full
