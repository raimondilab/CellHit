import sys
sys.path.append('./../..')

import pickle
from CellHit.search_and_inference import search, inference
from CellHit.data import DatasetLoader,prepare_data
from sklearn.metrics import mean_squared_error
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

drugID = 1373
dataset = 'gdsc'
random_state = 0
gene_selection_mode = 'moa_primed'
use_external_datasets = False
use_dumped_loaders = False
data_path = Path('/home/fcarli/CellHit/data')

model_path = Path('/home/fcarli/CellHit/results/gdsc/search_and_inference/moa_primed/models') / f'{drugID}.pkl'

with open(model_path, 'rb') as f:
    best_params = pickle.load(f)['best_params']


mses = []
corrs = []

for random_state in tqdm(range(20)):


    data_dict = prepare_data(drugID, dataset, random_state, 
                            gene_selection_mode, 
                            use_external_datasets=use_external_datasets,
                            data_path=data_path,celligner_output_path='/home/sblotas/mober_sim/mober/mober_alligned_data.feather',
                            use_dumped_loaders=use_dumped_loaders)

    #join the two dictionaries
    inference_results = inference(best_params=best_params, refit=True, internal_inference=True, gene_selection_mode=gene_selection_mode, return_model=True,  **data_dict)
            
            
    drug_mean = data_dict['loader'].get_drug_mean(drugID)
    drug_std = data_dict['loader'].get_drug_std(drugID)


    tdf = pd.DataFrame()
    tdf['DrugName'] = [data_dict['loader'].get_drug_name(drugID)]*len(inference_results['predictions'])
    tdf['DrugID'] = [drugID]*len(inference_results['predictions'])
    tdf['Predictions'] = (inference_results['predictions']* drug_std) + drug_mean
    tdf['Actual'] = ((data_dict['test_Y'] * drug_std) + drug_mean).values
    tdf['DepMapID'] = data_dict['test_indexes']
    tdf['Seed'] = [random_state]*len(inference_results['predictions'])

    mses.append(mean_squared_error(tdf['Actual'],tdf['Predictions']))
    corrs.append(np.corrcoef(tdf['Actual'],tdf['Predictions'])[0,1])


print(f'Mean MSE: {np.mean(mses)}')
print(f'Mean Corr: {np.mean(corrs)}')

