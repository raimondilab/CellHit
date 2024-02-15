import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .indexed_array import IndexedArray
from .metadata_processing import obtain_metadata
#from .sampler import StratisfiedSampler


class DatasetLoader():

    def __init__(self,
                dataset='gdsc',
                data_path='metadata.csv',
                celligner_output_path='celligner_CCLE_TCGA.feather',
                use_external_datasets=False,
                samp_x_tissue=2,random_state=0):
        
        #set data input path
        self.data_path = Path(data_path)
        self.celligner_output_path = Path(celligner_output_path)
        
        #set random state
        self.random_state = random_state
        self.samp_x_tissue = samp_x_tissue

        #obtain metadata for the selected dataset
        self.metadata = obtain_metadata(dataset=dataset,path=self.data_path)

        #load all transcriptomic data from the Celligner output
        if use_external_datasets:
            self.all_transcriptomics_data = pd.read_feather(self.celligner_output_path)
            self.source_mapper = self.all_transcriptomics_data[['index','Source']]#.set_index('index')
            self.cell_lines_data = self.all_transcriptomics_data[self.all_transcriptomics_data['Source']=='CCLE'].drop(columns=['Source']).set_index('index')
            self.all_transcriptomics_data = self.all_transcriptomics_data.drop(columns=['Source']).set_index('index')
        else:
            all_transcriptomics_data = pd.read_feather(self.celligner_output_path)
            #subset on CCLE cell lines
            self.cell_lines_data = all_transcriptomics_data[all_transcriptomics_data['Source']=='CCLE'].drop(columns=['Source']).set_index('index')

        #save the genes names for future use
        self.genes = self.cell_lines_data.columns
        
        #Consider pairs only of cell lines for which we have transcriptomic data
        self.metadata = self.metadata[self.metadata['DepMapID'].isin(set(self.cell_lines_data.index))]
        #self.y_metadata_scaled = False

        #TODO: subset on the drug should be putted here for efficiency. We do it otherwise for reproducibility
        #if subset_on_drug is not None:
        #    self.metadata = self.metadata[self.metadata['DrugID']==int(subset_on_drug)]

        #utilities
        #TODO: create some attributes to facilitate name,id and index conversion


    def split_and_scale(self,drugID=None,val_split=True,val_random_state=0,use_external=False,scale_full_metadata=False,pre_scaling=True):

        ##Split train-test##
        #shuffle the data and take the first 2 samples for each tissue type as test set
        test_depmapIDs = set(self.metadata.sample(frac=1,random_state=self.random_state).groupby('OncotreeLineage').head(self.samp_x_tissue).reset_index()['DepMapID'].values)
        #obtain DepMapIDs for the train set
        train_depmapIDs = set(self.metadata['DepMapID'].values) - set(test_depmapIDs)

        #split the metadata
        self.meta_train = self.metadata[self.metadata['DepMapID'].isin(train_depmapIDs)]
        self.meta_test = self.metadata[self.metadata['DepMapID'].isin(test_depmapIDs)]

        #TODO: for overall correctness the scaling should be done below, after removing the validation set from the train set
        if pre_scaling:
            self._scale(train_depmapIDs,use_external=use_external)

        #TODO: subsetting should not be done here for efficiency. We do it otherwise for reproducibility of legacy code
        if drugID is not None:
            self.meta_train = self.meta_train[self.meta_train['DrugID']==int(drugID)]
            self.meta_test = self.meta_test[self.meta_test['DrugID']==int(drugID)]

            #if val_split:
            #    self.meta_valid = self.meta_valid[self.meta_valid['DrugID']==int(drugID)]

        if val_split:
            #shuffle the data and take the first 2 samples for each tissue type as validation set
            valid_depmapIDs = set(self.meta_train.sample(frac=1,random_state=val_random_state).groupby('OncotreeLineage').head(self.samp_x_tissue).reset_index()['DepMapID'])
            #obtain DepMapIDs for the train set
            train_depmapIDs = set(self.meta_train['DepMapID'].values) - set(valid_depmapIDs)

            #split the metadata again
            self.meta_valid = self.meta_train[self.meta_train['DepMapID'].isin(valid_depmapIDs)]
            self.meta_train = self.meta_train[self.meta_train['DepMapID'].isin(train_depmapIDs)]
        
        if not pre_scaling:
            self._scale(train_depmapIDs,use_external=use_external)
       
        #Apply Y scaling and formatting
        self.meta_train['Y'] = self.meta_train.apply(lambda x: (x['Y'] - self.drug_mean_dict[x['DrugID']])/self.drug_std_dict[x['DrugID']],axis=1)
        self.meta_test['Y'] = self.meta_test.apply(lambda x: (x['Y'] - self.drug_mean_dict[x['DrugID']])/self.drug_std_dict[x['DrugID']],axis=1)
        
        if val_split:
            self.meta_valid['Y'] = self.meta_valid.apply(lambda x: (x['Y'] - self.drug_mean_dict[x['DrugID']])/self.drug_std_dict[x['DrugID']],axis=1)

        #If we want to scale the full metadata
        if scale_full_metadata:
            self.scaled_metadata = self.metadata.copy()
            self.scaled_metadata['Y'] = self.metadata.apply(lambda x: (x['Y'] - self.drug_mean_dict[x['DrugID']])/self.drug_std_dict[x['DrugID']],axis=1)

        #Prepare output
        train_X = self.Xs[list(self.meta_train['DepMapID'].values)]
        train_X = pd.DataFrame(train_X,columns=self.genes,index=self.meta_train['DepMapID'].values)

        test_X = self.Xs[list(self.meta_test['DepMapID'].values)]
        test_X = pd.DataFrame(test_X,columns=self.genes,index=self.meta_test['DepMapID'].values)

        train_Y = pd.Series(self.meta_train['Y'].values,index=self.meta_train['DrugID'].values)
        test_Y = pd.Series(self.meta_test['Y'].values,index=self.meta_test['DrugID'].values)

        out_values = [train_X,train_Y,test_X,test_Y]

        if val_split:
            valid_X = self.Xs[list(self.meta_valid['DepMapID'].values)]
            valid_X = pd.DataFrame(valid_X,columns=self.genes,index=self.meta_valid['DepMapID'].values)
            valid_Y = pd.Series(self.meta_valid['Y'].values,index=self.meta_valid['DrugID'].values)
            out_values += [valid_X,valid_Y]
    
        if use_external:
            #otain external data (source not CCLE)
            external_ids = list(self.source_mapper[self.source_mapper['Source']!='CCLE']['index'].values)
            external_X = self.Xs[external_ids]
            external_X = pd.DataFrame(external_X,columns=self.genes,index=external_ids)
            out_values += [external_X]

        return out_values
    

    def _scale(self,train_depmapIDs,use_external=False):

        #if not self.cell_scaled:
        #compute mean and std for each gene
        self.cell_mean = self.cell_lines_data[self.cell_lines_data.index.isin(train_depmapIDs)].mean()
        self.cell_std = self.cell_lines_data[self.cell_lines_data.index.isin(train_depmapIDs)].std()
        self.cell_lines_data = (self.cell_lines_data - self.cell_mean)/self.cell_std
        
        #X scaling and formatting everything
        if use_external:
            self.all_transcriptomics_data = (self.all_transcriptomics_data - self.cell_mean)/self.cell_std
            all_lines_dict = {cid:np.array(cell).reshape(1,-1) for cid,cell in zip(self.all_transcriptomics_data.index,self.all_transcriptomics_data.values)}
            self.Xs = IndexedArray(all_lines_dict)

        else:
            cell_lines_dict = {cid:np.array(cell).reshape(1,-1) for cid,cell in zip(self.cell_lines_data.index,self.cell_lines_data.values)}
            self.Xs = IndexedArray(cell_lines_dict)
        
        ##Y scaling and formatting##
        #compute mean and std for each drug
        self.drug_mean_dict = self.meta_train[['DrugID','Y']].groupby('DrugID').mean()
        self.drug_mean_dict = pd.Series(data=self.drug_mean_dict['Y'].values,index=self.drug_mean_dict.index).to_dict()
        self.drug_std_dict = self.meta_train[['DrugID','Y']].groupby('DrugID').std()
        self.drug_std_dict = pd.Series(data=self.drug_std_dict['Y'].values,index=self.drug_std_dict.index).to_dict()

    #define some getter methods
    def get_genes(self):
        return self.genes
    
    def get_drugs(self):
        return self.metadata['DrugID'].unique()
    
    def get_drugs_names(self):
        return self.metadata['Drug'].unique()
    
    def get_drug_name(self,drugID):

        #if first call, create the dictionary
        if not hasattr(self,'drug_name_dict'):
            mapping_data = self.metadata[['Drug','DrugID']].drop_duplicates()
            mapping_data['DrugID'] = mapping_data['DrugID'].astype(int)
            self.drug_name_dict = pd.Series(data=mapping_data['Drug'].values,index=mapping_data['DrugID']).to_dict()
        
        return self.drug_name_dict[int(drugID)]
    
    def get_drug_id(self,drug_name):

        #if first call, create the dictionary
        if not hasattr(self,'drug_id_dict'):
            mapping_data = self.metadata[['Drug','DrugID']].drop_duplicates()
            mapping_data['DrugID'] = mapping_data['DrugID'].astype(int)
            self.drug_id_dict = pd.Series(data=mapping_data['DrugID'].values,index=mapping_data['Drug']).to_dict()
        
        return self.drug_id_dict[drug_name]

