import argparse
import pickle
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import sys
sys.path.append('./../../')
from CellHit.data import obtain_metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gdsc", choices=["gdsc", "prism"])
    parser.add_argument("--model_type", type=str, default="moa_primed", choices=["moa_primed", "all_genes"])
    parser.add_argument("--data_path", type=str, default="./../../data/")
    parser.add_argument("--results_path", type=str, default="./../../results/")
    parser.add_argument("--output_path", type=str, default="./../../results/summaries")
    
    args = parser.parse_args()
    dict_args = vars(args)

    dict_args["data_path"] = Path(args.data_path)
    dict_args["models_path"] = Path(args.results_path) / f'{dict_args["dataset"]}/search_and_inference/{dict_args["model_type"]}/models/'
    
    #obtain metadata both for gdsc and prism
    data = obtain_metadata(dataset=dict_args["dataset"], path=dict_args["data_path"])

    #obtain drug ids
    drugs = data[['DrugID', 'Drug']].drop_duplicates()

    #read the models one by one and store the predictions
    result_df = []

    for drugID, drug in tqdm(drugs.values):

        model_path = dict_args["models_path"] / f"{drugID}.pkl"
        with open(model_path, "rb") as f:
            tres = pickle.load(f)['drug_results']
        
        tres = tres.rename(columns={"Drug": "DrugID"})
        tres['Drug'] = drug

        #group by Drug and DrugID and compute min,mean,median,max,std for both MSE and Corr
        tres = tres.groupby(['Drug', 'DrugID']).agg({'MSE': ['min', 'mean', 'median', 'max', 'std'], 'Corr': ['min', 'mean', 'median', 'max', 'std']}).reset_index()
        #flatten the multiindex
        tres.columns = ['_'.join(col).strip() for col in tres.columns.values]
        #rename the columns
        tres = tres.rename(columns={"Drug_": "Drug", "DrugID_": "DrugID"})
        #store the predictions
        result_df.append(tres)

    #concatenate all the results
    result_df = pd.concat(result_df)
    #save the results
    result_df.to_csv(Path(args.output_path) / f'{args.dataset}_{args.model_type}_performance_summary.csv', index=False)
        



