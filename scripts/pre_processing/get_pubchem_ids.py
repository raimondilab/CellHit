import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm
from pathlib import Path

import sys
sys.path.append('./../../')

from CellHit.data import get_pubchem_id,get_gdsc_drugs_metadata,get_prism_lfc_drugs_metadata

def map_gdsc(data_path):
    
    metadata = get_gdsc_drugs_metadata(data_path=data_path)
    
    #strategy using pubchempy and osme manual curation
    dname = []
    internal_id = []
    id = []
    id_type = []
    
    for did, drug in tqdm(zip(metadata['DrugID'].values,metadata['Drug'].values)):
    
        results = get_pubchem_id(drug)
    
        if results is not None:
    
            pbid,tpbid = results
            id.append(pbid)
            id_type.append(tpbid)
    
            dname.append(drug)
            internal_id.append(did)
        
    #manual add nutlin-3a
    dname.append('Nutilin-3a (-)')
    internal_id.append(1047)
    id.append(11433190)
    id_type.append('Compound')
    
    #manual add Bleomycin (50 uM)
    dname.append('Bleomycin (50 uM)')
    internal_id.append(1378)
    id.append(5360373)
    id_type.append('Compound')
    
    drug_ids = pd.DataFrame()
    drug_ids['Drug'] = dname
    drug_ids['DrugID'] = internal_id
    drug_ids['PubChemId'] = id
    drug_ids['id_type'] = id_type
    
    return drug_ids.dropna()

def map_prism(data_path):

    #exploit mapping from BROADID to PubChemID of the original Corsello et al. paper 
    repurposing_samples = pd.read_csv(data_path/'metadata'/'repurposing_samples_20200324.txt', sep='\t')[['broad_id','pubchem_cid']].dropna()
    repurposing_samples = repurposing_samples.rename(columns={'broad_id':'BroadID','pubchem_cid':'PubChemId'})

    metadata = get_prism_lfc_drugs_metadata(data_path=data_path)
    metadata['BroadID'] = metadata['BroadID'].apply(lambda x: x.split(':')[1])

    #merge with repurposing_samples
    metadata = pd.merge(repurposing_samples,metadata,on='BroadID',how='left').dropna()
    metadata['id_type'] = 'Compound'

    return metadata[['Drug','DrugID','PubChemId','id_type']]


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='all')
    argparser.add_argument('--data_path', type=str, default='./../../data')
    args = argparser.parse_args()

    data_path = Path(args.data_path)

    if args.dataset == 'gdsc':
        gdsc_mappings = map_gdsc(data_path=data_path)

        with open(data_path/'metadata'/'gdsc_pubchem_mappings.csv', 'w') as f:
            gdsc_mappings.to_csv(f, index=False)


    elif args.dataset == 'prism':
        prism_mappings = map_prism(data_path=data_path)

        with open(data_path/'metadata'/'prism_pubchem_mappings.csv', 'w') as f:
            prism_mappings.to_csv(f, index=False)


    elif args.dataset == 'all':
        gdsc_mappings = map_gdsc(data_path=data_path)
        prism_mappings = map_prism(data_path=data_path)

        with open(data_path/'metadata'/'gdsc_pubchem_mappings.csv', 'w') as f:
            gdsc_mappings.to_csv(f, index=False)

        with open(data_path/'metadata'/'prism_pubchem_mappings.csv', 'w') as f:
            prism_mappings.to_csv(f, index=False)

    

