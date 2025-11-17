import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

# --------------------------------------------
# 🔍 Smart Target Column Detector
# --------------------------------------------
POSSIBLE_TARGETS = [
    "isfraud", "fraud", "class", "label",
    "fraud_reported", "target", "is_fraud"
]

def detect_target_column(df):
    cols = df.columns.str.lower()
    for t in POSSIBLE_TARGETS:
        for col in df.columns:
            if col.lower() == t:
                return col
    return None


# --------------------------------------------
# 📂 Auto-detect datasets
# --------------------------------------------
def find_datasets():
    dataset_dir = "./data"
    datasets = {}
    for f in os.listdir(dataset_dir):
        if f.endswith(".csv"):
            key = f.replace(".csv", "").lower()  # e.g., "creditcard"
            datasets[key] = os.path.join(dataset_dir, f)
    return datasets


# --------------------------------------------
# 📂 Auto-detect models
# --------------------------------------------
def find_models():
    models = {}
    for f in os.listdir("./"):
        if f.endswith(".pkl"):
            key = f.replace(".pkl", "").lower()
            models[key] = f
    return models


# --------------------------------------------
# 🧠 Match model filenames to dataset keywords
# --------------------------------------------
def detect_domain_from_model(model_name, datasets):
    for domain in datasets.keys():
        if domain in model_name:   # e.g. credit_card_xgb.pkl contains "credit_card"
            return domain
    return None


# --------------------------------------------
# 🧪 Evaluate Model
# --------------------------------------------
def evaluate_model(model_path, dataset_path):
    try:
        print(f"\nEvaluating model: {model_path}")
        print(f"Dataset: {dataset_path}")

        model = pickle.load(open(model_path, "rb"))
        df = pd.read_csv(dataset_path)

        df = df.dropna(axis=1, how='all').dropna()

        target_col = detect_target_column(df)
        if not target_col:
            print("❌ No target column found. Skipping.")
            return None

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode categorical
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_pred = model.predict(X_test)
            y_prob = y_pred

        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5

        score = round((f1 + auc) / 2, 4)

        print(f"✔ F1: {f1:.4f}, AUC: {auc:.4f}, Weight: {score:.4f}")
        return score

    except Exception as e:
        print(f"❌ Error evaluating {model_path}: {e}")
        return None


# --------------------------------------------
# 🚀 MAIN EXECUTION
# --------------------------------------------
def main():
    datasets = find_datasets()
    models = find_models()

    final_weights = {}

    for model_key, model_path in models.items():
        domain = detect_domain_from_model(model_key, datasets)

        if not domain:
            print(f"⚠ No dataset matched for: {model_key}")
            continue

        dataset_path = datasets[domain]
        score = evaluate_model(model_path, dataset_path)

        if score is not None:
            final_weights[model_key] = score

    # Save weights JSON
    with open("model_weights.json", "w") as f:
        json.dump(final_weights, f, indent=4)

    print("\n🎉 Saved final model weights to model_weights.json")
    print(final_weights)


if __name__ == "__main__":
    main()
