import sys
sys.path.append('./../../')
import os
from CellHit.search_and_inference import search, inference
from CellHit.data import DatasetLoader,prepare_data
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
from AsyncDistribJobs.operations import add_jobs, get_jobs_by_state, job_statistics,fetch_job
from AsyncDistribJobs.operations import process_job


#define the pipline to run hyperparameter search and inference on the test sets of the different random states
def search_and_inference_drug(drugID, dataset,  
                              gene_selection_mode, 
                              use_external_datasets, 
                              data_path, celligner_output_path,
                              random_state=0,
                              n_iter=20,
                              save_output=True, results_path=None,
                              use_dumped_loaders=False,
                              **kwargs):
    
    #assert that data is provided


    #if we are saving the output, we need to provide the results path
    if save_output:
        assert results_path is not None, "If save_output is True, results_path must be provided"
        results_path = Path(results_path)
        assert results_path.exists(), f"results_path not found at {results_path}"


    #obtain the data
    data_dict = prepare_data(drugID, dataset, random_state, 
                        gene_selection_mode, 
                        use_external_datasets=use_external_datasets,
                        data_path=data_path,celligner_output_path=celligner_output_path,
                        use_dumped_loaders=use_dumped_loaders)
    
    #lunch the hyperparameter search procedure
    best_params, study = search(n_trials=kwargs['n_trials'], n_startup_trials=kwargs['n_startup_trials'],
                                cv_iterations=kwargs['cv_iterations'], num_parallel_tree=kwargs['num_parallel_tree'], gpuID=kwargs['gpuID'], random_state=random_state,
                                gene_selection_mode=gene_selection_mode,
                                **data_dict)

    #run inference on the test set, but first initialize the output container
    predictions_df = []
    output_container = {}
    output_container['HyperparamStudy'] = study
    output_container['BestParams'] = best_params
    output_container['Models'] = []
    output_container['Means'] = []
    output_container['Stds'] = []
    if gene_selection_mode == 'moa_primed':
        output_container['Genes'] = data_dict['genes']
    output_container['DrugName'] = data_dict['loader'].get_drug_name(drugID)
    
    mses = []
    corrs = []

    for random_state in range(n_iter):

        data_dict = prepare_data(drugID, dataset, random_state, 
                        gene_selection_mode, 
                        use_external_datasets=use_external_datasets,
                        data_path=data_path,celligner_output_path=celligner_output_path,
                        use_dumped_loaders=use_dumped_loaders)

        #join the two dictionaries
        inference_results = inference(best_params=best_params, refit=True, internal_inference=True, gene_selection_mode=gene_selection_mode, return_model=True,  **data_dict)
        
        output_container['Models'].append(inference_results['model'])
        
        drug_mean = data_dict['loader'].get_drug_mean(drugID)
        drug_std = data_dict['loader'].get_drug_std(drugID)

        output_container['Means'].append(drug_mean)
        output_container['Stds'].append(drug_std)

        tdf = pd.DataFrame()
        tdf['DrugName'] = [data_dict['loader'].get_drug_name(drugID)]*len(inference_results['predictions'])
        tdf['DrugID'] = [drugID]*len(inference_results['predictions'])
        tdf['Predictions'] = (inference_results['predictions']* drug_std) + drug_mean
        tdf['Actual'] = ((data_dict['test_Y'] * drug_std) + drug_mean).values
        tdf['DepMapID'] = data_dict['test_indexes']
        tdf['Seed'] = [random_state]*len(inference_results['predictions'])
        predictions_df.append(tdf)

        mses.append(mean_squared_error(tdf['Actual'],tdf['Predictions']))
        corrs.append(np.corrcoef(tdf['Actual'],tdf['Predictions'])[0,1])


    predictions_df = pd.concat(predictions_df)

    drug_results = pd.DataFrame()
    drug_results['DrugID'] = [drugID]*len(mses)
    drug_results['Iteration'] = list(range(len(mses)))
    drug_results['MSE'] = mses
    drug_results['Corr'] = corrs

    output_container['DrugResults'] = drug_results
    output_container['RawPredictions'] = predictions_df

    if save_output:

        #save ouptut container
        with open(results_path/dataset/'search_and_inference'/gene_selection_mode/'models'/f'{drugID}.pkl', 'wb') as f:
            pickle.dump(output_container, f)
        
        #save predictions as csv
        predictions_df.to_csv(results_path/dataset/'search_and_inference'/gene_selection_mode/'raw_internal_predictions'/f'{drugID}.csv',index=False)
    
    return predictions_df, output_container#, study

#function to check whether the provided paths exist / are valid
def sanity_checks(data_path, celligner_output_path,
                  search_database_path=None,
                  search_mode = 'single_drug',
                  save_output=False, results_path=None,
                  dataset='gdsc', gene_selection_mode='all_genes',
                  use_dumped_loaders=False,
                  **kwargs):
    
    data_path = Path(data_path)
    celligner_output_path = Path(celligner_output_path)

    assert data_path.exists(), f"data_path not found at {data_path}"
    assert celligner_output_path.exists(), f"celligner_output_path not found at {celligner_output_path}"
    
    if search_mode == 'full_asynch':
        search_database_path = Path(search_database_path)
        assert search_database_path.exists(), f"search_database_path not found at {search_database_path}"

    if use_dumped_loaders:
        assert (data_path/'loader_dumps'/dataset).exists(), f"dumped_loaders folder not found at {data_path/dataset/'dumped_loaders'}"

    if save_output:
        results_path = Path(results_path)
        assert results_path.exists(), f"results_path not found at {results_path}"
        assert (results_path/dataset/'search_and_inference'/gene_selection_mode/'models').exists(), f"output folder for model containers does not exist at {results_path/dataset/'search_and_inference'/gene_selection_mode}"
        assert (results_path/dataset/'search_and_inference'/gene_selection_mode/'raw_internal_predictions').exists(), f"output folder for raw internal predictions does not exist at {results_path/dataset/'search_and_inference'/gene_selection_mode/'raw_internal_predictions'}"



def run_full_asynch_search(args,search_database_path):

    engine = create_engine(f'sqlite:///{search_database_path/args["dataset"]}_{args["gene_selection_mode"]}_search_database.db')
    configure_database(engine,reset=args['build_search_db'])

    #if specified, build the search database
    if args['build_search_db']:

        if args['use_dumped_loaders']:
            with open(Path(args['data_path'])/'loader_dumps'/args['dataset']/f'0.pkl','rb') as f:
                loader = pickle.load(f)

        else:

            #populate the search database
            loader = DatasetLoader(dataset=args['dataset'],
                                    data_path=args['data_path'],
                                    celligner_output_path=args['celligner_output_path'],
                                    use_external_datasets=False,
                                    samp_x_tissue=2,
                                    random_state=0)
        
        drugs_ids = loader.get_drugs_ids()

        #for drugID in tqdm(drugs_ids):
        #    add_job(payload={'drugID': int(drugID)},cid=f'{drugID}')

        jobs_list = [Job(state='pending',payload={'drugID': int(drugID)},cid=f'{drugID}') for drugID in drugs_ids]
        add_jobs(jobs_list)

    while len(get_jobs_by_state('pending')) > 0:
        process_job(search_and_inference_drug,**args)


def run_single_drug(args):
    predictions_df, output_container = search_and_inference_drug(args)
    

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--drugID', type=int, default=1909)
    argparser.add_argument('--celligner_output_path', type=str, default='./../../data/transcriptomics/celligner_CCLE_TCGA.feather')
    argparser.add_argument('--data_path', type=str, default='./../../data/')
    argparser.add_argument('--dataset', type=str, default='gdsc')
    argparser.add_argument('--gene_selection_mode', type=str, default='all_genes')
    argparser.add_argument('--num_parallel_tree', type=int, default=5)
    argparser.add_argument('--gpuID', type=int, default=0)
    argparser.add_argument('--n_trials', type=int, default=300)
    argparser.add_argument('--n_startup_trials', type=int, default=100)
    argparser.add_argument('--random_state', type=int, default=0)
    argparser.add_argument('--n_iter', type=int, default=20)
    argparser.add_argument('--use_external_datasets', default=False, action='store_true')
    argparser.add_argument('--cv_iterations', type=int, default=3)
    argparser.add_argument('--use_dumped_loaders', default=False, action='store_true')
    argparser.add_argument('--val_random_state', type=int, default=0)
    argparser.add_argument('--disable_saving', default=False, action='store_true')
    argparser.add_argument('--results_path', type=str, default='./../../results/')

    argparser.add_argument('--search_mode', type=str, default='single_drug')
    argparser.add_argument('--build_search_db', default=False, action='store_true')
    #argparser.add_argument('--search_database_path', type=str, default='./../../results/gdsc/search_database')


    args = argparser.parse_args()
    args.save_output = not args.disable_saving

    args_dict = vars(args)
    
    sanity_checks(**args_dict)

    if args.search_mode == 'asynch':

        if 'drugID' in args_dict:
            del args_dict['drugID']
            
        search_database_paths = Path(f'./../../results/{args.dataset}/search_database')
        run_full_asynch_search(args_dict, search_database_paths)

    elif args.search_mode == 'single_drug':
        data = search_and_inference_drug(**args_dict)




