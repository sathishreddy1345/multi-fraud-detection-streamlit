
import pickle
import pandas as pd
from fraud_modules.aggregator import combine_model_scores

model_files = {
    'rf': 'models/randomforest_model.pkl',
    'xgb': 'models/catboost_model.pkl',
    'lgbm': 'models/lightgbm_model.pkl',
    'cat': 'models/catboost_model.pkl',
    'lr': 'models/logisticregression_model.pkl',
    'iso': 'models/isolationforest_model.pkl'
}

models = {}
for key, path in model_files.items():
    with open(path, 'rb') as f:
        models[key] = pickle.load(f)

def predict_creditcard_fraud(input_df):
    scores = {}
    for name, model in models.items():
        if name == 'iso':
            scores[name] = -model.decision_function(input_df)[0]
        else:
            scores[name] = model.predict_proba(input_df)[0][1]
    combined = combine_model_scores(scores)
    return combined, scores
