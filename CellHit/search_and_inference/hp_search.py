from ..models import AutoXGBRegressor
from ..data import DatasetLoader, GeneGetter

import pandas as pd


def search(n_trials=300, n_startup_trials=100,
           cv_iterations=3, num_parallel_tree=5, gpuID=0, random_state=0,
           gene_selection_mode='moa_primed',
           cv_data=None,train_X=None, train_Y=None, valid_X=None, valid_Y=None,
           *args, **kwargs):

    # Prepare data based on gene selection mode
    if gene_selection_mode == 'moa_primed':
        #assert that data is provided
        assert cv_data is not None, "If gene_selection_mode is moa_primed, data must be provided"
    else: 
        #assert that train and validation data is provided
        assert (train_X is not None) and (train_Y is not None) and (valid_X is not None) and (valid_X is not None), "If gene_selection_mode is not moa_primed, X_train, y_train, X_val and y_val must be provided"
        

    # Model fitting
    xgb = AutoXGBRegressor(num_parallel_tree=num_parallel_tree, gpuID=gpuID)

    #perform hyperparameter search
    if gene_selection_mode == 'moa_primed':
        xgb.search(cv_data=cv_data, n_trials=n_trials, n_startup_trials=n_startup_trials)
    else:
        xgb.search(train_X=train_X, train_Y=train_Y, valid_X=valid_X, valid_Y=valid_Y, 
                   n_trials=n_trials, n_startup_trials=n_startup_trials)

    # Select best parameters (same as before)
    best_params = max(xgb.study.best_trials, key=lambda t: t.values[1]).params
    best_params['device'] = f'cuda:{gpuID}'
    best_params['tree_method'] = 'hist'
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'
    best_params['num_parallel_tree'] = num_parallel_tree

    #if gene_selection_mode == 'moa_primed':
    #    return best_params, genes, xgb.study
    #else:
    return best_params, xgb.study

