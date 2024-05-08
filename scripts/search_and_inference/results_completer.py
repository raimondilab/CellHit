import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from pandarallel import pandarallel


import sys
import pickle
sys.path.append('./../../')
from CellHit.data import DatasetLoader, get_gdsc_drugs_metadata,get_prism_lfc_drugs_metadata
from CellHit.utils import QuantileScoreComputer, FaissKNN


def complete_drug(drugID,external_dataset,**kwargs):

    #read predictions
    predictions = pd.read_csv(kwargs['results_path']/f'full_inference_{external_dataset}/{drugID}.csv')
    predictions = predictions[predictions['Source'] == external_dataset]

    #read shaps
    shaps = pd.read_csv(kwargs['results_path']/f'full_inference_shaps_{external_dataset}/{drugID}.csv')
    shaps = shaps



def compute_stats(**dict_args):

    #load metadata
    loader = DatasetLoader(dataset=dict_args['dataset'],
                            data_path=dict_args['data_path'],
                            celligner_output_path=dict_args['data_path'] / dict_args['transcr_suffix'],
                            use_external_datasets=dict_args['use_external_datasets'],
                            samp_x_tissue=2,
                            random_state=0)
    
    #compute the min, median and max for each drug (needed to contextualize predictions)
    stats = loader.metadata[['DrugID','Y']].groupby(['DrugID']).agg(['min','median','max']).reset_index()

    #get rid of multiindex
    stats.columns = ['DrugID','min','median','max']

    return stats


def assemble_predictions(drug_indexes,topk=15,**dict_args):

    #define the path in case of external datasets or inference on tcga
    if dict_args['use_external_datasets']:
        inference_path = dict_args['results_path'] / dict_args['dataset'] / 'search_and_inference' / 'moa_primed' / f'full_inference_{dict_args["external_dataset"]}'
        shaps_path = dict_args['results_path'] / dict_args['dataset'] / 'search_and_inference' / 'moa_primed' / f'full_inference_shaps_raw_{dict_args["external_dataset"]}'
        prefix = 'IEO_' if dict_args['external_dataset'] == 'PDAC' else 'FPS_'
    else:
        inference_path = dict_args['results_path'] / dict_args['dataset'] / 'search_and_inference' / 'moa_primed' / 'full_inference'
        shaps_path = dict_args['results_path'] / dict_args['dataset'] / 'search_and_inference' / 'moa_primed' / 'full_inference_shaps_raw'

    assemble_predictions = []

    #loop over the drugs in order to assemble the predictions and the shap values for each prediction
    for drug in tqdm(drug_indexes):

        #read the predictions for a given drug
        predictions = pd.read_csv(inference_path / f'{drug}.csv')
        
        #load the shap data
        with open(shaps_path / f'{drug}.pkl', 'rb') as file:
            shaps = pickle.load(file)
        values = shaps.values
        #we transform into numpy arrays to use advanced indexing later
        feature_names = np.array(shaps.feature_names)
        

        #take topk 15 absolute values for each instance (most important features)
        topk_indexes = np.abs(values).argsort(axis=1)[:,-topk:]
        #transform the indexes into feature names
        topk_feature_names = feature_names[topk_indexes]
        #transform the bidimensional array into a list of lists
        topk_feature_names = topk_feature_names.tolist()

        #create a new column with the topk feature names
        predictions['TopGenes'] = topk_feature_names
        predictions['TopGenes'] = predictions['TopGenes'].apply(lambda x: ','.join(x))

        #append the predictions to the list
        assemble_predictions.append(predictions)

    #concatenate the predictions
    predictions = pd.concat(assemble_predictions)

    #compure quantile score
    qs = QuantileScoreComputer(predictions,cell_col='Index',drug_col='DrugID',score_col='Predictions')
    if dict_args['use_pandarallel']:
        predictions['QuantileScore'] = predictions.parallel_apply(lambda x: qs.compute_score(drug=x['DrugID'],cell=x['Index'],score=x['Predictions']),axis=1)
    else:
        predictions['QuantileScore'] = predictions.apply(lambda x: qs.compute_score(drug=x['DrugID'],cell=x['Index'],score=x['Predictions']),axis=1)

    #NEIGHBORS (in response space)
    predictions_ccle = predictions[predictions['Source'] == 'CCLE']
    predictions_ccle = predictions_ccle[['Index','DrugID','Predictions']].set_index('Index').pivot(columns='DrugID',values='Predictions')
    ccle_neighs = FaissKNN(predictions_ccle)

    predictions_tcga = predictions[predictions['Source'] == 'TCGA']
    predictions_tcga = predictions_tcga[~predictions_tcga['Index'].str.contains('THR')]
    predictions_tcga = predictions_tcga[~predictions_tcga['Index'].str.contains('TH')]
    predictions_tcga = predictions_tcga[~predictions_tcga['Index'].str.contains('TARGET')]
    predictions_tcga = predictions_tcga[['Index','DrugID','Predictions']].set_index('Index').pivot(columns='DrugID',values='Predictions')
    tcga_neighs = FaissKNN(predictions_tcga)

    #Subset on the external dataset if needed (only after computing the quantile score, as all data is needed)
    if dict_args['use_external_datasets']:
        predictions = predictions[predictions['Source'] == dict_args['external_dataset']]

    query_predictions = predictions[['Index','DrugID','Predictions']].set_index('Index').pivot(columns='DrugID',values='Predictions')

    #find neighs in ccle
    ccle_neighs_df = ccle_neighs.knn(query_predictions, k=1).rename(columns={'neighbour_point':'response_CCLE_neigh','query_point':'Index'})[['response_CCLE_neigh','Index']]

    #find neighs in tcga
    tcga_neighs_df = tcga_neighs.knn(query_predictions, k=1).rename(columns={'neighbour_point':'response_TCGA_neigh','query_point':'Index'})[['response_TCGA_neigh','Index']]

    #TODO: add metadata to the neighbors

    predictions = predictions.merge(ccle_neighs_df,on='Index').merge(tcga_neighs_df,on='Index')

    return predictions


def compute_transcriptomic_neighbors(**dict_args):

    trascr_data = pd.read_feather(dict_args['data_path'] / dict_args['transcr_suffix'])
    ccle_trascr = trascr_data[trascr_data['Source'] == 'CCLE'].set_index('index').drop('Source',axis=1)
    ccle_neighs = FaissKNN(ccle_trascr)
    tcga_trascr = trascr_data[trascr_data['Source'] == 'TCGA'].set_index('index').drop('Source',axis=1)
    tcga_neighs = FaissKNN(tcga_trascr)

    if dict_args['use_external_datasets']:
        query = trascr_data[trascr_data['Source'] == dict_args['external_dataset']].set_index('index').drop('Source',axis=1)

    else:
        query = trascr_data.set_index('index').drop('Source',axis=1)

    #find neighs in ccle
    ccle_neighs_df = ccle_neighs.knn(query, k=1).rename(columns={'neighbour_point':'transcr_CCLE_neigh','query_point':'Index'})[['transcr_CCLE_neigh','Index']]
    tcga_neighs_df = tcga_neighs.knn(query, k=1).rename(columns={'neighbour_point':'transcr_TCGA_neigh','query_point':'Index'})[['transcr_TCGA_neigh','Index']]

    #merge the neighbors
    trascr_neighs = ccle_neighs_df.merge(tcga_neighs_df,on='Index')

    return trascr_neighs


def add_cell_and_tcga_meta(drugs_predictions,**dict_args):

    #--TCGA--
    tcga_clinical = pd.read_csv(dict_args['data_path']/'metadata/tcga_clinical.tsv', sep='\t')[['case_submitter_id','site_of_resection_or_biopsy','primary_diagnosis']]
    diagnosis_mapper = pd.Series(tcga_clinical['primary_diagnosis'].values,index=tcga_clinical['case_submitter_id']).to_dict()
    site_mapper = pd.Series(tcga_clinical['site_of_resection_or_biopsy'].values,index=tcga_clinical['case_submitter_id']).to_dict()

    drugs_predictions['transcr_TCGA_neigh'] = drugs_predictions['transcr_TCGA_neigh'].apply(lambda x: x[:-3])
    drugs_predictions['response_TCGA_neigh'] = drugs_predictions['response_TCGA_neigh'].apply(lambda x: x[:-3])

    #add diagnosis and site of resection
    drugs_predictions['transcr_TCGA_neigh_diagnosis'] = drugs_predictions['transcr_TCGA_neigh'].map(diagnosis_mapper)
    drugs_predictions['transcr_TCGA_neigh_site'] = drugs_predictions['transcr_TCGA_neigh'].map(site_mapper)

    drugs_predictions['response_TCGA_neigh_diagnosis'] = drugs_predictions['response_TCGA_neigh'].map(diagnosis_mapper)
    drugs_predictions['response_TCGA_neigh_site'] = drugs_predictions['response_TCGA_neigh'].map(site_mapper)

    #--CCLE--
    model_data = pd.read_csv(dict_args['data_path']/'metadata/Model.csv')
    model_data_mapper = pd.Series(model_data['OncotreePrimaryDisease'].values,index=model_data['ModelID']).to_dict()
    cell_line_name_mapper = pd.Series(model_data['CellLineName'].values,index=model_data['ModelID']).to_dict()

    drugs_predictions['transcr_CCLE_neigh_Oncotree'] = drugs_predictions['transcr_CCLE_neigh'].map(model_data_mapper)
    drugs_predictions['response_CCLE_neigh_Oncotree'] = drugs_predictions['response_CCLE_neigh'].map(model_data_mapper)

    drugs_predictions['transcr_CCLE_neigh_CellLineName'] = drugs_predictions['transcr_CCLE_neigh'].map(cell_line_name_mapper)
    drugs_predictions['response_CCLE_neigh_CellLineName'] = drugs_predictions['response_CCLE_neigh'].map(cell_line_name_mapper)

    return drugs_predictions


def get_targets(merged,**dict_args):

    #get_gdsc_drugs_metadata,get_prism_lfc_drugs_metadata

    if dict_args['dataset'] == 'gdsc':
        target_mapper =  get_gdsc_drugs_metadata(data_path=dict_args['data_path'])[['DrugID','repurposing_target']].dropna() 
    else:
        target_mapper =  get_prism_lfc_drugs_metadata(data_path=dict_args['data_path'])[['DrugID','repurposing_target']].dropna()

    target_mapper = pd.Series(target_mapper['repurposing_target'].values,index=target_mapper['DrugID']).to_dict()

    merged['PutativeTarget'] = merged['DrugID'].map(target_mapper)

    def target_checker(x):
        #first of all check whether x is None
        if not isinstance(x['PutativeTarget'],str):
            return None
        else:
            #obtain the list of putative targets by splitting on commma and stripping
            putative_targets = [i.strip() for i in x['PutativeTarget'].split(',')]

            #obtain the set of putative targets
            top_genes_set = set([i.strip() for i in x['TopGenes'].split(',')])

            recovered_targets = []

            for target in putative_targets:
                if target in top_genes_set:
                    recovered_targets.append(target)

            if len(recovered_targets) == 0:
                return None
            else:
                return ','.join(recovered_targets)

    #target must exists and be in the set of TopGenes
    merged['RecoveredTarget'] = merged.apply(target_checker,axis=1)

    return merged


def add_glioblastoma_metadata(merged,**dict_args):

    merged['temp_index'] = merged['Index'].apply(lambda x: x[4:])

    glioblastoma_metadata = pd.read_excel(dict_args['data_path']/'metadata'/'FPS_metadata.xlsx')
    glioblastoma_metadata = glioblastoma_metadata[['GBSample_code','Primary P/ Recurrence R','NADH FLIM TMZ Response']]
    glioblastoma_metadata = glioblastoma_metadata.rename(columns={'GBSample_code':'SampleID','Primary P/ Recurrence R':'Recurrence','NADH FLIM TMZ Response':'Response'})

    merged = merged.merge(glioblastoma_metadata,left_on='temp_index',right_on='SampleID',how='left')

    merged = merged.drop(columns=['temp_index','SampleID'],axis=1)

    return merged


def add_pancreatic_metadata(merged,**dict_args):

    merged['temp_index'] = merged['Index'].apply(lambda x: x[4:])

    pancreatic_metadata = pd.read_csv(dict_args['data_path']/'metadata'/'LMD_RNAseq_annotation.txt',sep='\t')

    merged = merged.merge(pancreatic_metadata,left_on='temp_index',right_on='Sample',how='left')
    merged = merged.drop(columns=['temp_index','Sample'],axis=1)

    return merged


def add_external_metadata(merged,**dict_args):

    if dict_args['external_dataset'] == 'GBM':
        merged = add_glioblastoma_metadata(merged,**dict_args)
    else:
        merged = add_pancreatic_metadata(merged,**dict_args)

    return merged


def column_renamer(merged):

    renamer_dict = {
                    'Index':'SampleIndex',
                    'Stds':'PredictionsStd',
                    'min':'ExperimentalMin',
                    'median':'ExperimentalMedian',
                    'max':'ExperimentalMax',
                    'MSE_median':'ModelMSE',
                    'Corr_median':'ModelCorr',
                    'TopGenes':'TopLocalShapGenes',
                    }
    
    merged = merged.rename(columns=renamer_dict)

    return merged

def columns_resort(merged,**dict_args):

    if dict_args['external_dataset'] == 'GBM':
        specific_meta = ['Recurrence','Response']
    elif dict_args['external_dataset'] == 'PDAC':
        specific_meta = ['Biotype','Morphology','Patient_ID']

    cols = ['DrugName','DrugID','Source','SampleIndex']

    if dict_args['use_external_datasets']:
        cols += specific_meta

    cols += ['Predictions',\
            'PredictionsStd',\
            'QuantileScore',\
            'ExperimentalMin',\
            'ExperimentalMedian',\
            'ExperimentalMax',\
            'ModelMSE',\
            'ModelCorr',\
            'transcr_CCLE_neigh',\
            'transcr_CCLE_neigh_CellLineName',\
            'transcr_CCLE_neigh_Oncotree',\
            'response_CCLE_neigh',\
            'response_CCLE_neigh_CellLineName',\
            'response_CCLE_neigh_Oncotree',\
            'transcr_TCGA_neigh',\
            'transcr_TCGA_neigh_diagnosis',\
            'transcr_TCGA_neigh_site',\
            'response_TCGA_neigh',\
            'response_TCGA_neigh_diagnosis',\
            'response_TCGA_neigh_site',\
            'PutativeTarget',\
            'TopLocalShapGenes',\
            'RecoveredTarget']
    
    return merged[cols]

    

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Gathers the results obtained during inference')
    argparser.add_argument('--dataset', type=str, default='gdsc',choices=['gdsc','prism'])
    argparser.add_argument('--external_dataset', type=str, default='GBM',choices=['PDAC','GBM','None'])
    #argparser.add_argument('--results_path', type=str, default='../../results/gdsc/search_and_inference/moa_primed/')
    argparser.add_argument('--results_path', type=str, default='../../results/')
    argparser.add_argument('--data_path', type=str, default='../../data/')
    argparser.add_argument('--use_pandarallel', type=bool, default=True)
    argparser.add_argument('--n_workers', type=int, default=8)

    args = argparser.parse_args()
    dict_args = vars(args)

    if args.use_pandarallel:
        pandarallel.initialize(progress_bar=True,nb_workers=args.n_workers)

    dict_args['results_path'] = Path(args.results_path)
    #dict_args['inference_path'] = dict_args['results_path'] / dict_args['dataset'] / 'search_and_inference' / 'moa_primed'
    dict_args['data_path'] = Path(args.data_path)

    if args.external_dataset != 'None':
        dict_args['use_external_datasets'] = True
        dict_args['transcr_suffix'] = f'transcriptomics/celligner_CCLE_TCGA_{dict_args["external_dataset"]}.feather'
    else:
        dict_args['use_external_datasets'] = False
        dict_args['transcr_suffix'] = 'transcriptomics/celligner_CCLE_TCGA.feather'

    #compute stats (in a dataframe, min, median, max for each drugID)
    stats = compute_stats(**dict_args)
    drug_indexes = list(stats['DrugID'].values)
    
    #assemble predictions for each drug
    predictions_with_genes = assemble_predictions(drug_indexes=drug_indexes,**dict_args)

    #compute the transcriptomic neighbors
    trascr_neigh = compute_transcriptomic_neighbors(**dict_args)

    #load drug performances
    drug_performances = pd.read_csv(dict_args['results_path'] / 'summaries' / f'{dict_args["dataset"]}_moa_primed_performance_summary.csv')[['DrugID','MSE_median','Corr_median']]
    drug_performances = drug_performances.rename(columns={'MSE_mean':'MSE','Corr_mean':'Corr'})

    #merge everything
    merged = predictions_with_genes.merge(stats,on='DrugID').merge(drug_performances,on='DrugID').merge(trascr_neigh,on='Index')

    #add metadata to the neighbors
    merged = add_cell_and_tcga_meta(merged,**dict_args)

    #recovered putative targets
    merged = get_targets(merged,**dict_args)

    #add metadata from the external dataset
    if dict_args['use_external_datasets']:
        merged = add_external_metadata(merged,**dict_args)

    #rename columns
    merged = column_renamer(merged)

    #reorder columns
    merged = columns_resort(merged,**dict_args)

    #save the results
    merged.to_csv(dict_args['results_path'] / 'summaries' / f'full_results_{dict_args["dataset"]}_{dict_args["external_dataset"]}.csv',index=False)

    
    #merged.to_csv('partial_results.csv')
    #TODO: add metadata both to transcriptomic and response space neighbors (from ccle and tcga)







