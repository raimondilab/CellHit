import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer

def corr_metric(y_true, y_pred):
    score = np.corrcoef(y_true, y_pred)[0, 1]
    return score

def mse_metric(y_true, y_pred):
    score = mean_squared_error(y_true, y_pred)
    return score

# Create scoring dictionary
scoring = {
    'corr': make_scorer(corr_metric),
    'mse': make_scorer(mse_metric)
}