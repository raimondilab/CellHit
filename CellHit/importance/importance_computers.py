from .metrics import corr_metric, mse_metric
from .shufflers import batcher, batcher_numba

import pickle

from ..data import DatasetLoader

import gc

import pickle

import xgboost as xgb
import shap

import numpy as np
import pandas as pd

from tqdm.auto import tqdm


def permutation_importance(temp_X,temp_Y,
                           genes,
                           model,
                           important_genes_idxs,
                           chunk_size=200000,
                           random_state=0,
                           use_numba=True):
    
    if use_numba:
        batched = batcher_numba(temp_X, np.array(important_genes_idxs), random_state=random_state)
    else:
        batched = batcher(temp_X, important_genes_idxs, random_state=random_state)
    
    #if batched.shape[0] becomes too large, the GPU will run out of memory, chunck predictions
    if batched.shape[0] > chunk_size:
            
        #split in chunks of 200000 rows
        n_chunks = int(np.ceil(batched.shape[0]/chunk_size))
        total_predictions = batched.shape[0]  # Total number of predictions to be made

        # Pre-allocate a NumPy array for predictions
        predictions = np.empty((total_predictions,), dtype=np.float32)

        # Fill in the pre-allocated array with predictions
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_predictions)
            predictions[start_idx:end_idx] = model.predict(xgb.DMatrix(batched[start_idx:end_idx])).astype(np.float32)

    else:
        #predictions = model.predict(xgb.DMatrix(batched)).astype(np.float32)
        predictions = model.predict(pd.DataFrame(columns=genes,data=batched))['predictions']#.astype(np.float32)

# 1. Unpack the predictions
    n_samples = temp_X.shape[0]
    # The first n_samples predictions are the original predictions
    orig_predictions = predictions[:n_samples]
    # Split the shuffled predictions into separate arrays for each feature
    shuffled_predictions = np.split(predictions[n_samples:], len(important_genes_idxs))

    # Initialize containers for performance drops
    corr_deltas = []
    perc_corr_deltas = []
    mse_deltas = []
    perc_mse_deltas = []

    # 2. Evaluate the original model performance
    orig_corr = corr_metric(temp_Y, orig_predictions)
    orig_mse = mse_metric(temp_Y, orig_predictions)

    # 3. Calculate performance drops for each feature
    for i, feat_idx in enumerate(important_genes_idxs):
        # Shuffled predictions for this feature
        shuf_preds = shuffled_predictions[i]

        # Evaluate performance on shuffled data
        shuf_corr = corr_metric(temp_Y, shuf_preds)
        shuf_mse = mse_metric(temp_Y,     shuf_preds)

        # Compute performance drops
        corr_delta = shuf_corr - orig_corr
        mse_delta = shuf_mse - orig_mse

        # Store the performance drops
        corr_deltas.append(corr_delta)
        perc_corr_deltas.append(corr_delta / orig_corr)
        mse_deltas.append(mse_delta)
        perc_mse_deltas.append(mse_delta / orig_mse)

    feature_importance_df = pd.DataFrame({
        'genes': important_genes_idxs,
        'corr_delta': corr_deltas,
        'mse_delta': mse_deltas,
        'perc_corr_delta': perc_corr_deltas,
        'perc_mse_delta': perc_mse_deltas
    })

    return feature_importance_df


def shap_computer(X,booster,return_shap_explain_obj=False):
    
    #compute the shap values
    explainer = shap.TreeExplainer(booster,data=X)
    
    if return_shap_explain_obj:
        shap_obj = explainer(X)
        shap_values = shap_obj.values
    
    else:
        shap_values = explainer.shap_values(X)
        
    shap_values = np.mean(np.abs(shap_values), axis=0).astype(np.float32)
    feature_shap_df = pd.DataFrame()
    feature_shap_df['genes'] = X.columns
    feature_shap_df['shap'] = shap_values
    
    if return_shap_explain_obj:
        return feature_shap_df,shap_obj
    else: 
        return feature_shap_df
    

def shap_objects_combiner(shap_objects):
    
    background_data = shap_objects[0].data
    
    #obtain a tridimensional array with the shap values
    shap_values = np.array([x.values for x in shap_objects])
    shap_values = np.mean(shap_values,axis=0)
    
    shap_base_values = np.array([x.base_values for x in shap_objects])
    shap_base_values = np.mean(shap_base_values,axis=0)
    
    explanation = shap.Explanation(values=shap_values,
                        base_values=shap_base_values,
                        data=background_data,
                        feature_names=shap_objects[0].feature_names)
    
    return explanation
    



def compute_feature_importance(drugID,loader, model,
                               return_shap_obj=False, 
                               n_permutations=3, chunk_size=200000, use_numba=True):
    # Get gene names
    genes = loader.get_genes()
    pos_to_genes_mapper = {idx: gene for idx, gene in enumerate(genes)}
    data_dict = loader.split_and_scale(drugID=drugID)

    # Find indexes for which training importance (gini criterion) is not 0
    gain_importances = model.get_important_features()
    # Get gene importance if it's in gain_importances otherwise 0
    gain_importances = np.array([gain_importances[i] if i in gain_importances else 0 for i in genes])
    important_genes_idxs = np.where(gain_importances > 0)[0]

    # Send to 32 bit to save memory
    temp_X = pd.concat([data_dict['valid_X'], data_dict['test_X']]).values.astype(np.float32)
    temp_Y = pd.concat([data_dict['valid_Y'], data_dict['test_Y']]).values.astype(np.float32)

    # Here we can compute SHAP values before deleting the data
    shap_values, shap_obj = shap_computer(pd.concat([data_dict['train_X'], data_dict['valid_X'], data_dict['test_X']]), model.model, return_shap_explain_obj=True)

    # Free memory (can be a lot of data)
    del data_dict
    del loader
    gc.collect()

    tmp_permutations = []

    for perm in range(n_permutations):
        # Compute permutation importance
        perm_imp = permutation_importance(temp_X, temp_Y,
                                          genes,
                                          model,
                                          important_genes_idxs,
                                          chunk_size=chunk_size,
                                          random_state=perm,
                                          use_numba=use_numba)

        perm_imp['genes'] = perm_imp['genes'].map(pos_to_genes_mapper)
        tmp_permutations.append(perm_imp)

    # Concatenate the results and average across the permutations
    tmp_permutations = pd.concat(tmp_permutations).groupby('genes').mean().reset_index()
    features_df = pd.merge(shap_values,tmp_permutations,on='genes',how='left').fillna(0)

    if return_shap_obj:
        return features_df, shap_obj
    else:
        return features_df


