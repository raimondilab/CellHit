import sys
sys.path.append('./../../')
import gc
import os
from CellHit.search_and_inference import search, inference
from CellHit.data import DatasetLoader,prepare_data
from CellHit.importance import compute_feature_importance, shap_objects_combiner
from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

import pickle

from tqdm.auto import tqdm

from sqlalchemy import create_engine
from AsyncDistribJobs.models import Job
from AsyncDistribJobs.operations import configure_database
from AsyncDistribJobs.operations import add_jobs, get_jobs_by_state
from AsyncDistribJobs.operations import process_job

def compute_importance(drugID,
                       dataset,
                       models_path=None,
                       importance_output_path=None,
                       use_dumped_loaders=True,
                       data_path=None,
                       n_permutations=3,
                       use_numba=True,
                       chunk_size=200000,**kwargs):
    
    #load the models for a given drug
    with open(Path(models_path)/f'{drugID}.pkl', 'rb') as f:
        models = pickle.load(f)['Models'] 

    #initialize the containers for importance dataframes and shap objects
    raw_feature_importances = []
    shap_objs = [] 
        
    for random_state,model in tqdm(enumerate(models),total=len(models)):

        #Obtain the loader
        if use_dumped_loaders:
            with open(data_path/'loader_dumps'/dataset/f'{random_state}.pkl', 'rb') as f:
                loader = pickle.load(f)

        else:
            loader = DatasetLoader(dataset,
                                   data_path, 
                                   random_state=random_state)


        #compute the importance
        importance_df, shap_obj = compute_feature_importance(drugID=drugID, loader=loader, model=model, 
                                                          n_permutations=n_permutations, return_shap_obj=True,
                                                          use_numba=use_numba, chunk_size=chunk_size)
        
        importance_df['seed'] = random_state
        
        #append the importance to the list
        raw_feature_importances.append(importance_df)
        shap_objs.append(shap_obj)
        

    #raw dataframe containing the importance disaggregated by seed
    raw_feature_importances = pd.concat(raw_feature_importances)
    #obtain combined shap object
    shap_explainer = shap_objects_combiner(shap_objs)
    #obrain the aggregated importance
    averaged_feature_importances = raw_feature_importances.groupby('genes').mean().reset_index().sort_values(by='corr_delta',ascending=False)
    
    #save the disaggregated and aggregated importance
    raw_feature_importances.to_csv(importance_output_path/'raw_feature_importances'/f'{drugID}.csv',index=False)
    averaged_feature_importances.to_csv(importance_output_path/'averaged_feature_importances'/f'{drugID}.csv',index=False)

    #save the shap object
    with open(importance_output_path/'shap_explainers'/f'{drugID}.pkl','wb') as f:
        pickle.dump(shap_explainer,f)


def run_full_async_importance(args, importance_database_path):
    engine = create_engine(f'sqlite:///{importance_database_path}/{args["dataset"]}_importance_database.db')
    configure_database(engine, reset=args['build_importance_db'])

    # if specified, build the importance database
    if args['build_importance_db']:

        if args['use_dumped_loaders']:
            with open(Path(args['data_path'])/'loader_dumps'/args['dataset']/f'0.pkl', 'rb') as f:
                loader = pickle.load(f)
        else:
            # populate the importance database
            loader = DatasetLoader(use_external_datasets=False,
                                   samp_x_tissue=2,
                                   random_state=0,**args)

        drugs_ids = loader.get_drugs_ids()

        jobs_list = [Job(state='pending', payload={'drugID': int(drugID)}, cid=f'{drugID}') for drugID in drugs_ids]
        add_jobs(jobs_list)



    while len(get_jobs_by_state('pending')) > 0:
        process_job(compute_importance, **args)


#def run_single_drug_importance(args):
    #compute_importance(args_dict)
    #importance_df.to_csv(importance_database_path/f'{args.dataset}_{args.gene_selection_mode}_importance.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the hyperparameter search for the CellHit model')
    parser.add_argument('--drugID', type=str, default=1909, help='ID of the drug')
    parser.add_argument('--dataset', type=str,default='gdsc',help='Name of the dataset')
    parser.add_argument('--data_path', type=str,default='./../../data', help='Path to the data folder')
    parser.add_argument('--use_dumped_loaders', action='store_true', help='Use pre-dumped data loaders')
    #TODO: add the possibility to run importance on MOA primed models
    #parser.add_argument('--gene_selection_mode', type=str, help='Mode of gene selection')
    parser.add_argument('--celligner_output_path', type=str, default='./../../data/transcriptomics/celligner_CCLE_TCGA.feather', help='Path to the Celligner output')
    parser.add_argument('--n_permutations', type=int, default=3, help='Number of permutations to use for the permutation importance')
    parser.add_argument('--no_numba', action='store_true', help='Do not use numba for the permutation importance')
    parser.add_argument('--chunk_size', type=int, default=200000, help='Chunk size for the permutation importance')

    #parser.add_argument('--importance_database_path', type=str, help='Path to the importance database')
    parser.add_argument('--importance_mode', type=str, default='async', help='Mode of importance computation')
    parser.add_argument('--build_importance_db', action='store_true', help='Build a new importance database')

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['use_numba'] = not args.no_numba
    args_dict['data_path'] = Path(args.data_path)
    args_dict['models_path'] = Path(f'./../../results/{args.dataset}/search_and_inference/all_genes/models')
    args_dict['importance_output_path'] = Path(f'./../../results/{args.dataset}/importance')
    args_dict['celligner_output_path'] = Path(args.celligner_output_path)


    if args.importance_mode == 'single_drug':
        compute_importance(**args_dict)

    elif args.importance_mode == 'async':

        #if drugID is not None, delete it from the args_dict
        if 'drugID' in args_dict:
            del args_dict['drugID']
        importance_database_paths = Path(f'./../../results/{args.dataset}/importance_database')
        run_full_async_importance(args_dict, importance_database_paths)





#/home/fcarli/CellHit/results/gdsc/search_and_inference/all_genes/models