import sys
sys.path.append('./../../')

import pandas as pd

from pathlib import Path

from CellHit.search_and_inference import inference
from CellHit.data import DatasetLoader, prepare_data

from sqlalchemy import create_engine
from AsyncDistribJobs.models import Job
from AsyncDistribJobs.operations import configure_database
from AsyncDistribJobs.operations import add_jobs, get_jobs_by_state, job_statistics,fetch_job
from AsyncDistribJobs.operations import process_job

import argparse
import pickle


def external_inference(drugID,
                       dataset=None,
                       external_dataset=None,
                       results_path=None,
                       models_path=None,
                       **kwargs):

    #obtain the best parameters from the search object
    with open(models_path/f'{int(drugID)}.pkl', 'rb') as f:
        best_params = pickle.load(f)['best_params']

    data_dict = prepare_data(use_external_datasets=True,
                             gene_selection_mode='moa_primed',
                             **args)
    
    results = inference(best_params=best_params,
                        refit=True, internal_inference=False, 
                        gene_selection_mode='moa_primed',
                        return_shaps=True, 
                        return_model=True,  **data_dict)
    
    #path adjustments
    full_inference = 'full_inference' + f'_{external_dataset}' if external_dataset is not None else 'full_inference'
    full_inference_shaps = 'full_inference_shaps' + f'_{external_dataset}' if external_dataset is not None else 'full_inference_shaps'
    
    #create the dataframe of the predictions
    predictions_df = pd.DataFrame()
    predictions_df['DrugName'] = [data_dict['loader'].get_drug_name(drugID)]*len(results['predictions'])
    predictions_df['DrugID'] = [drugID]*len(results['predictions'])
    predictions_df['Index'] = data_dict['external_indexes']
    predictions_df['Predictions'] = results['predictions']
    predictions_df['Source'] = data_dict['loader'].get_indexes_sources(data_dict['external_indexes'].tolist())
    predictions_df.to_csv(Path(results_path) / dataset / 'search_and_inference' / 'moa_primed' / full_inference / f'{drugID}.csv',index=False)

    #create the dataframe of the shap values
    shap_df = pd.DataFrame()
    #values are inside results['shap_values'].values, whereas columns are inside results['shap_values'].feature_names
    shap_df['Index'] = data_dict['external_indexes']
    shap_df['Source'] = data_dict['loader'].get_indexes_sources(data_dict['external_indexes'].tolist())
    shap_df = pd.concat([shap_df, pd.DataFrame(results['shap_values'].values, columns=results['shap_values'].feature_names)], axis=1)
    shap_df = pd.DataFrame(results['shap_values'].values, columns=results['shap_values'].feature_names)
    shap_df.to_csv(Path(results_path) / dataset / 'search_and_inference' / 'moa_primed' / full_inference_shaps / f'{drugID}.csv',index=False)


def run_full_asynch_inference(args,search_database_path):

    engine = create_engine(f'sqlite:///{search_database_path/args.dataset}_{args.external_dataset}_inference_database.db')
    configure_database(engine,reset=args.build_inference_db)

    #if specified, build the search database
    if args.build_inference_db:

        if args.use_dumped_loaders:
            with open(Path(args.data_path)/'loader_dumps'/f'{args.dataset}_inference'/f'{args.external_dataset}.pkl','rb') as f:
                loader = pickle.load(f)

        else:

            #populate the search database
            loader = prepare_data(use_external_datasets=True,
                             gene_selection_mode='moa_primed',
                             **args)
        
        drugs_ids = loader.get_drugs_ids()

        #for drugID in drugs_ids:
        #    add_job(payload={'drugID': int(drugID)},cid=f'{drugID}')

        jobs_list = [Job(state='pending',payload={'drugID': int(drugID)},cid=f'{drugID}') for drugID in drugs_ids]
        add_jobs(jobs_list)


    while len(get_jobs_by_state('pending')) > 0:
        process_job(external_inference,**vars(args))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Inference for CellHit')
    argparser.add_argument('--dataset', type=str, default='gdsc')
    argparser.add_argument('--drugID', type=int, default=1909)
    argparser.add_argument('--random_state', type=int, default=0)
    argparser.add_argument('--celligner_output_path', type=str, default='./../../data/transcriptomics/celligner_CCLE_TCGA')
    argparser.add_argument('--use_dumped_loaders', default=False, action='store_true')
    argparser.add_argument('--external_dataset', type=str, default='GBM')
    argparser.add_argument('--data_path', type=str, default='./../../data/')
    argparser.add_argument('--results_path', type=str, default='./../../results/')
    argparser.add_argument('--gpu_id', type=int, default=0)

    argparser.add_argument('--inference_mode', type=str, default='full_asynch')
    argparser.add_argument('--build_inference_db', default=False, action='store_true')
    argparser.add_argument('--inference_database_path', type=str, default='./../../results/gdsc/search_database')

    args = argparser.parse_args()

    args_dict = vars(args)
    args_dict['models_path'] = Path(args.results_path) / args.dataset /'search_and_inference' / 'moa_primed' / 'models'
    args_dict['data_path'] = Path(args.data_path)

    if args.external_dataset is not None:
        args_dict['celligner_output_path'] = Path(args.celligner_output_path + f'_{args.external_dataset}.feather')
    else:
        args_dict['celligner_output_path'] = Path(args.celligner_output_path+'.feather')


    if args.inference_mode == 'full_asynch':
        search_database_paths = Path(f'./../../results/{args.dataset}/search_database')
        run_full_asynch_inference(args, search_database_paths)
    else:
        external_inference(args_dict)

