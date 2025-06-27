
import pickle
import pandas as pd
from fraud_modules.aggregator import combine_model_scores

model_files = {
    'rf': 'models/paysim_rf.pkl',
    'xgb': 'models/paysim_xgb.pkl',
    'lgbm': 'models/paysim_lgbm.pkl',
    'cat': 'models/paysim_cat.pkl',
    'lr': 'models/paysim_lr.pkl',
    'iso': 'models/paysim_iso.pkl'
}

models = {}
for key, path in model_files.items():
    with open(path, 'rb') as f:
        models[key] = pickle.load(f)

def predict_paysim_fraud(input_df):
    scores = {}
    for name, model in models.items():
        if name == 'iso':
            scores[name] = -model.decision_function(input_df)[0]
        else:
            scores[name] = model.predict_proba(input_df)[0][1]
    combined = combine_model_scores(scores)
    return combined, scores, input_df
