from ..data import prepare_data
from ..models import EnsembleXGBoost, CustomXGBoost

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

#import xgboost as xgb


def inference(model = None,
              train_X=None,train_Y=None,valid_X=None,valid_Y=None,
              cv_data=None,
              test_X=None,
              external_X=None,
              best_params=None,
              refit=False,
              internal_inference=True,
              return_model=False,
              gene_selection_mode='all_genes',**kwargs):
    
    #this uses an already trained model
    if not refit:
        assert model is not None, "If refit is False, a model must be provided"

    #this trains a new model, usefull in case of newly alligned data
    if refit:
        assert best_params is not None, "best_params must be provided"

    #if we are refitting, we need to provide the data
    if (refit) and (gene_selection_mode == 'all_genes'):
        assert (train_X is not None) and (train_Y is not None) and (valid_X is not None) and (valid_Y is not None)
        model = all_genes_inference_refit(train_X,train_Y,valid_X,valid_Y,best_params)

    elif (refit) and (gene_selection_mode == 'moa_primed'):
        assert (cv_data is not None)
        model = knowledge_primed_inference_refit(cv_data,best_params)

    out_dict = {}

    #make predictions
    if internal_inference:
        out_dict['predictions'] = model.predict(test_X)
    else:
        out_dict['predictions'] = model.predict(external_X)

    #return the model if requested
    if return_model:
        out_dict['model'] = model
    
    return out_dict



def all_genes_inference_refit(train_X,train_Y,valid_X,valid_Y,best_params,random_state=0):
    model = CustomXGBoost(base_params=best_params)
    model.fit(train_X, train_Y, valid_X, valid_Y)
    return model

def knowledge_primed_inference_refit(cv_data,best_params):
    model = EnsembleXGBoost(base_params=best_params)
    model.fit(cv_data)
    return model

