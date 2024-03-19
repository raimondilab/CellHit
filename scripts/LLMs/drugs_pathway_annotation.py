import pandas as pd
import json
import requests
from pathlib import Path
from tqdm.auto import tqdm
import argparse
from collections import defaultdict, Counter

import sys
sys.path.append('./../../')

from CellHit.data import get_reactome_layers, get_gdsc_drugs_metadata,get_prism_lfc_drugs_metadata,get_genes_pathways


def target_in_pathway_criterion(metadata, reactome_pathways):

    genes_sets = reactome_pathways['Genes'].values

    #initialize a default dictionary with empty set as default value
    target_pathways = defaultdict(set)

    for id,drug in zip(metadata['DrugID'].values,metadata['repurposing_target'].values):

        if isinstance(drug,str):
            targets = set(drug.split(','))

            for gset in genes_sets:

                if len(targets & gset) > 0:
                    target_pathways[id].update(gset)
                
        #target_pathways[id] = pths

    return dict(target_pathways)


def LLMs_selection_criterion(metadata, reactome_pathways, pathway_selection_path, self_consistency_threshold=2):

    pathways_to_genes = pd.Series(reactome_pathways['Genes'].values,index=reactome_pathways['PathwayName']).to_dict()

    LLM_pathways = defaultdict(set)
    
    #for json files inside the pathway_selection_path    
    for drug in pathway_selection_path.iterdir():
        
        if drug.suffix == '.json':

            drug_id = drug.stem
            
            with open(drug) as f:
                pathway_selection = json.load(f)

            #apply the self consistency threshold
            selected_pathways = [key for key in pathway_selection if pathway_selection[key]['count'] >= self_consistency_threshold]

            for pathway in selected_pathways:
                LLM_pathways[drug_id].update(pathways_to_genes[pathway])

    return dict(LLM_pathways)

def ligand_annotation_criterion(reactome_drugs,reactome_pathways,dataset,data_path):

    if dataset == 'gdsc':
        mappings = pd.read_csv(data_path/'metadata'/'gdsc_pubchem_mappings.csv')
    elif dataset == 'prism':
        mappings =pd.read_csv(data_path/'metadata'/'prism_pubchem_mappings.csv')

    reactome_pathways = pd.Series(reactome_pathways['Genes'].values,index=reactome_pathways['PathwayID']).to_dict()

    def gene_getter(path_list):
        genes = set()
        for path in path_list:
            genes = genes.union(reactome_pathways.get(path,set()))
        return genes

    pubchem_to_drugid = pd.Series(mappings['DrugID'].values,index=mappings['PubChemId']).to_dict()
    reactome_drugs['DrugID'] = reactome_drugs['PubChemID'].apply(lambda x: pubchem_to_drugid.get(x,None))
    reactome_drugs = reactome_drugs.dropna()[['DrugID','PathwayID']]
    reactome_drugs['DrugID'] = reactome_drugs['DrugID'].astype(int)
    reactome_drugs = reactome_drugs.groupby('DrugID')['PathwayID'].apply(set).reset_index()
    reactome_drugs['Genes'] = reactome_drugs['PathwayID'].apply(gene_getter)

    return pd.Series(reactome_drugs['Genes'].values,index=reactome_drugs['DrugID']).to_dict()
   
    

def most_common_pathways(reactome_pathways,pathway_selection_path,data_path,most_common=15):

    most_common_genes = set()
    pathway_to_genes = pd.Series(reactome_pathways['Genes'].values,index=reactome_pathways['PathwayName']).to_dict()

    pathways = []

    for drug in pathway_selection_path.iterdir():

        if drug.suffix == '.json':

            with open(drug) as f:
                pathway_selection = json.load(f)


            selected_pathways = [key for key in pathway_selection if pathway_selection[key]['count'] >= 2]

            pathways.extend(selected_pathways)

    pathways = Counter(pathways).most_common(most_common)

    for pathway,_ in pathways:
        most_common_genes.update(pathway_to_genes[pathway])

    return most_common_genes



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get drugs pathways')
    parser.add_argument('--dataset', type=str, help='Dataset to use', default='gdsc')
    parser.add_argument('--data_path', type=str, help='Path to the data', default='./../../data/')
    parser.add_argument('--results_path', type=str, help='Path to the results', default='./../../results/')
    parser.add_argument('--reactome_layer', type=int, help='Reactome layer to use', default=1)
    #parser.add_argument('--pathway_selection_path', type=str, help='Path to the pathway selection json files', default='./../../results/LLMs/pathway_selection/')
    parser.add_argument('--self_consistency_threshold', type=int, help='Threshold for self consistency', default=2)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    pathway_selection_path = Path(args.results_path) / args.dataset / 'reactome_paths'

    if args.dataset == 'gdsc':
        metadata = get_gdsc_drugs_metadata(data_path)
    elif args.dataset == 'prism':
        metadata = get_prism_lfc_drugs_metadata(data_path)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    reactome_pathways = pd.read_csv(data_path/'reactome'/'pathways_to_genes.csv')
    reactome_drugs = pd.read_csv(data_path/'reactome'/'pathways_to_drugs.csv')
    #transform strings into sets
    reactome_pathways['Genes'] = reactome_pathways['Genes'].apply(lambda x: eval(x))
    
    target_in_pathway_dict = target_in_pathway_criterion(metadata, reactome_pathways)
    ligand_annotation_dict = ligand_annotation_criterion(reactome_drugs,reactome_pathways,args.dataset,data_path)
    LLM_selection_dict = LLMs_selection_criterion(metadata, reactome_pathways, pathway_selection_path, args.self_consistency_threshold)

    if args.dataset == 'gdsc':
        most_common_pathway_genes = most_common_pathways(reactome_pathways,pathway_selection_path,data_path,most_common=15)
        with open(data_path/'MOA_data'/'gdsc_most_common_genes.txt','w') as f:
            f.write('\n'.join(most_common_pathway_genes))

    with open(data_path/'MOA_data'/f'{args.dataset}_target_drugID_to_genes.json','w') as f:
        target_in_pathway_dict = {str(k):list(v) for k,v in target_in_pathway_dict.items()}
        json.dump(target_in_pathway_dict,f)

    with open(data_path/'MOA_data'/f'{args.dataset}_ligand_drugID_to_genes.json','w') as f:
        ligand_annotation_dict = {str(k):list(v) for k,v in ligand_annotation_dict.items()}
        json.dump(ligand_annotation_dict,f)

    with open(data_path/'MOA_data'/f'{args.dataset}_LLM_drugID_to_genes.json','w') as f:
        LLM_selection_dict = {str(k):list(v) for k,v in LLM_selection_dict.items()}
        json.dump(LLM_selection_dict,f)

