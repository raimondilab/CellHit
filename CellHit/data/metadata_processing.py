import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def obtain_gdsc(data_path,drug_threshold=10,**kwargs):
                
    #if drug_threshold is specified in kwargs, use that value
    #if 'drug_threshold' in kwargs:
    #    drug_threshold = kwargs['drug_threshold']
        
    #define path for the data
    path = Path(data_path)

    #read responses from GDSC dataset
    data = pd.read_csv(path/'metadata'/'GDSC2_fitted_dose_response_24Jul22.csv')
    data = data.rename(columns={'COSMIC_ID': 'COSMICID','DRUG_ID':'DrugID','DRUG_NAME':'Drug','LN_IC50':'Y'})
    data = data[['COSMICID','DrugID','Drug','Y']]

    #read cell lines metadata
    cell_lines = pd.read_csv(path/'metadata'/'Model.csv')
    cell_lines = cell_lines[['ModelID', 'COSMICID', 'OncotreeCode', 'OncotreeSubtype','OncotreePrimaryDisease', 'OncotreeLineage']]

    #merge the two pieces of information
    data = data.merge(cell_lines, on='COSMICID', how='inner').rename(columns={'ModelID':'DepMapID'})
    
    #filter on drugs with id in GDSC_drugs.csv
    drugs = set(pd.read_csv(path/'metadata'/'GDSC_drugs.csv')['DRUG_ID'].values)
    data = data[data['DrugID'].isin(drugs)]

    count_drug = data['DrugID'].value_counts()
    count_drug = set(count_drug[count_drug>drug_threshold].index.tolist())
    data = data[data['DrugID'].isin(count_drug)]
    
    return data
    
    
def obtain_prism_lfc(data_path,drug_threshold=10,**kwargs):
    
    path = Path(data_path)
    
    model = pd.read_csv(path/'metadata'/'Model.csv')
    model = model[['ModelID', 'COSMICID', 'OncotreeCode', 'OncotreeSubtype','OncotreePrimaryDisease', 'OncotreeLineage']]
    model = model.rename(columns={'ModelID':'DepMapID'})
    
    prism_lfc = pd.read_csv(path/'metadata'/'Repurposing_Public_23Q2_Extended_Primary_Data_Matrix.csv')
    mapping_metadata = pd.read_csv(path/'metadata'/'Repurposing_Public_23Q2_Extended_Primary_Compound_List.csv')
    mapping_metadata = mapping_metadata[['Drug.Name','IDs']]
    mapping_metadata.columns = ['Drug','DrugID']
    
    prism_lfc.columns = ['DrugID'] + [i for i in prism_lfc.columns[1:]]
    prism_lfc = pd.merge(prism_lfc,mapping_metadata,on='DrugID',how='inner')
    #melt prism_lfc
    prism_lfc = pd.melt(prism_lfc,id_vars=['DrugID','Drug'],var_name='DepMapID',value_name='Y').dropna(subset=['Y'])
    prism_lfc = prism_lfc.dropna()
    
    broad_to_name = pd.Series(prism_lfc['Drug'].values,index=prism_lfc['DrugID']).to_dict()
    
    #everything that has the same DrugID and same DepMapID should be collapsed into one row
    
    #collapse same DrugID and DepMapID
    prism_lfc = prism_lfc[['DrugID','DepMapID','Y']].groupby(['DrugID','DepMapID']).mean().reset_index()
    prism_lfc['DrugName'] = prism_lfc['DrugID'].apply(lambda x: broad_to_name[x])
    name_to_broad = pd.Series(prism_lfc['DrugID'].values,index=prism_lfc['DrugName']).to_dict()
    
    
    #collapse same DrugName and DepMapID
    prism_lfc = prism_lfc[['DrugName','DepMapID','Y']].groupby(['DrugName','DepMapID']).mean().reset_index()
    prism_lfc['DrugID'] = prism_lfc['DrugName'].apply(lambda x: name_to_broad[x])
    
    prism_lfc = prism_lfc.rename(columns={'DrugName':'Drug','DrugID':'BroadID'})

    #create a numericID for each DrugID
    drug_to_id = pd.Series(range(len(prism_lfc['BroadID'].unique())),index=prism_lfc['BroadID'].unique()).to_dict()

    prism_lfc['DrugID'] = prism_lfc['BroadID'].apply(lambda x: drug_to_id[x])
    
    prism_lfc = pd.merge(prism_lfc,model,on='DepMapID',how='inner')
    
    return prism_lfc
    

def obtain_metadata(dataset='gdsc',path='./data',drug_threshold=10,**kwargs):

    if dataset == 'gdsc':
        return obtain_gdsc(path,drug_threshold,**kwargs)
    
    if dataset == 'prism':
        return obtain_prism_lfc(path,drug_threshold,**kwargs)

def obtain_drugs_metadata(dataset='gdsc',path='./data'):
    
    if dataset == 'gdsc':
        return get_gdsc_drugs_metadata(path)
    
    if dataset == 'prism':
        return get_prism_lfc_drugs_metadata(path)
    

def get_gdsc_drugs_metadata(data_path='./data'):
    data_path = Path(data_path)
    data = pd.read_csv(data_path/'metadata'/'GDSC_drugs.csv')
    data = data.rename(columns={'DRUG_ID':'DrugID','DRUG_NAME':'Drug','PATHWAY_NAME':'MOA','HGCN_TARGETS':'repurposing_target'})
    return data

    
def get_prism_lfc_drugs_metadata(data_path='./data'):
    data_path = Path(data_path)
    data = obtain_prism_lfc(data_path)
    drugs = data[['Drug','BroadID','DrugID']].drop_duplicates()
    prism_metadata = pd.read_csv(data_path /'metadata'/ 'Repurposing_Public_23Q2_Extended_Primary_Compound_List.csv')
    out = pd.merge(drugs,prism_metadata[['IDs','MOA','repurposing_target']],left_on='BroadID',right_on='IDs',how='left').drop(columns=['IDs'])
    out = out.drop_duplicates()
    return out



class GeneGetter():

    def __init__(self,dataset='gdsc',data_path=None,available_genes=None,**kwargs):

        self.dataset = dataset
        self.available_genes = set(available_genes)

        if dataset == 'gdsc':
            
            #read the most common genes
            with open(data_path/'MOA_data'/'gdsc_most_common_genes.txt','r') as f:
                self.common_genes = f.read().splitlines()
        
        #read LLM associated genes
        with open(data_path/'MOA_data'/f'{dataset}_LLM_drugID_to_genes.json','r') as f:
            self.llm_genes = json.load(f)

        #read ligand associated genes
        with open(data_path/'MOA_data'/f'{dataset}_ligand_drugID_to_genes.json','r') as f:
            self.ligand_genes = json.load(f)

        with open(data_path/'MOA_data'/f'{dataset}_target_drugID_to_genes.json','r') as f:
            self.target_genes = json.load(f)

    def get_genes(self,drugID):

        genes = []

        if str(drugID) in self.llm_genes.keys():
            genes.extend(self.llm_genes[str(drugID)])

        if (self.dataset == 'gdsc') and (str(drugID) not in self.llm_genes.keys()):
            genes.extend(self.common_genes)

        if str(drugID) in self.ligand_genes.keys():
            genes.extend(self.ligand_genes[str(drugID)])

        if str(drugID) in self.target_genes.keys():
            genes.extend(self.target_genes[str(drugID)])

        return list(set(genes).intersection(self.available_genes))



        
        



