
import pickle
import pandas as pd
from fraud_modules.aggregator import combine_model_scores

model_files = {
    'rf': 'models/vehicle_loan_rf.pkl',
    'xgb': 'models/vehicle_loan_xgb.pkl',
    'lgbm': 'models/vehicle_loan_lgbm.pkl',
    'cat': 'models/vehicle_loan_cat.pkl',
    'lr': 'models/vehicle_loan_lr.pkl',
    'iso': 'models/vehicle_loan_iso.pkl'
}

models = {}
for key, path in model_files.items():
    with open(path, 'rb') as f:
        models[key] = pickle.load(f)

def predict_vehicle_loan_fraud(input_df):
    scores = {}
    for name, model in models.items():
        if name == 'iso':
            scores[name] = -model.decision_function(input_df)[0]
        else:
            scores[name] = model.predict_proba(input_df)[0][1]
    combined = combine_model_scores(scores)
    return combined, scores, input_df
