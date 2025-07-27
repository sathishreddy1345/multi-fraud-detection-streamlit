# paysim.py

import numpy as np
import pandas as pd
import joblib

# Load models and pipelines
model_names = ["rf", "xgb", "cat", "lr", "iso"]
models = {}
models_full = {}

for name in model_names:
    model_path = f"models/paysim_{name}.pkl"
    try:
        pipeline = joblib.load(model_path)
        models[name] = pipeline.named_steps["clf"]
        models_full[name] = pipeline
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")

def predict_paysim_fraud(df: pd.DataFrame):
    global models_full
    df = df.copy()

    

    # Save target column if it exists
    if "isFraud" in df.columns:
        df = df.rename(columns={"isFraud": "actual"})
    if "actual" in df.columns:
        actual = df["actual"]
        df = df.drop("actual", axis=1)
    else:
        actual = None

    # Ensure numerical + categorical columns are properly handled
    df = df.select_dtypes(include=[np.number, "object"]).copy()
    df.fillna(0, inplace=True)

    # Use Random Forest model for main scoring
    default_model = models_full.get("rf") or list(models_full.values())[0]
    X_processed = default_model.named_steps["pre"].transform(df)

    # Prepare predictions
    predictions = {}
    for name, pipeline in models_full.items():
        model = pipeline.named_steps["clf"]
        try:
            X_transformed = pipeline.named_steps["pre"].transform(df)

            if name == "iso":
                preds = model.predict(X_transformed)
                scores = np.where(preds == -1, 1, 0)
                predictions[name] = np.mean(scores)
            else:
                scores = model.predict_proba(X_transformed)[:, 1]
                predictions[name] = np.mean(scores)
        except Exception as e:
            predictions[name] = 0
            print(f"⚠️ Prediction failed for model {name}: {e}")

    avg_score = np.mean(list(predictions.values()))

    # Return processed dataframe with correct feature names
    feature_names = default_model.named_steps["pre"].get_feature_names_out()
    processed_df = pd.DataFrame(X_processed, columns=feature_names)

    if actual is not None:
        processed_df["actual"] = actual.values

    return avg_score, predictions, processed_df


# For app integration
models = models
models_full = models_full
