def combine_model_scores(scores: dict):
    weights = {'rf': 0.2, 'xgb': 0.2, 'lgbm': 0.2, 'cat': 0.2, 'lr': 0.1, 'iso': 0.1}
    return sum(scores[m] * weights.get(m, 0) for m in scores)
