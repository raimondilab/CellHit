import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .indexed_array import IndexedArray
from .metadata_processing import obtain_metadata, GeneGetter
#from .sampler import StratisfiedSampler


class DatasetLoader():

    def __init__(self,
                dataset='gdsc',
                data_path='metadata.csv',
                celligner_output_path='celligner_CCLE_TCGA.feather',
                use_external_datasets=False,
                samp_x_tissue=2,random_state=0,**kwargs):
        
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

        #scalig flag for the transcriptomic data
        self.x_are_scaled = False

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

        #out_values = [train_X,train_Y,test_X,test_Y]
        out_values = {'train_X':train_X,'train_Y':train_Y,'test_X':test_X,'test_Y':test_Y}

        if val_split:
            valid_X = self.Xs[list(self.meta_valid['DepMapID'].values)]
            valid_X = pd.DataFrame(valid_X,columns=self.genes,index=self.meta_valid['DepMapID'].values)
            valid_Y = pd.Series(self.meta_valid['Y'].values,index=self.meta_valid['DrugID'].values)
            #out_values += [valid_X,valid_Y]
            out_values['valid_X'] = valid_X
            out_values['valid_Y'] = valid_Y
        
    
        if use_external:
            #otain external data (source not CCLE)
            #external_ids = list(self.source_mapper[self.source_mapper['Source']!='CCLE']['index'].values)
            external_ids = self.Xs.get_all_keys()
            external_X = self.Xs[external_ids]
            external_X = pd.DataFrame(external_X,columns=self.genes,index=external_ids)
            #out_values += [external_X]
            out_values['external_X'] = external_X

        return out_values
    

    def _scale(self,train_depmapIDs,use_external=False):
        
        #check if data is already scaled
        if self.x_are_scaled:
            self._revert_scaling(use_external=use_external)

        #if we are doing inference without external data, we scale the data using the mean and std of the train set
        if not use_external:
            self.cell_mean = self.cell_lines_data[self.cell_lines_data.index.isin(train_depmapIDs)].mean()
            self.cell_std = self.cell_lines_data[self.cell_lines_data.index.isin(train_depmapIDs)].std()
            self.cell_lines_data = (self.cell_lines_data - self.cell_mean)/self.cell_std
        
        #otherwise we scale the data using the mean and std of the full dataset since we will use the external data as "test"
        if use_external:
            self.cell_mean = self.cell_lines_data.mean()
            self.cell_std = self.cell_lines_data.std()
            self.cell_lines_data = (self.cell_lines_data - self.cell_mean)/self.cell_std
            self.all_transcriptomics_data = (self.all_transcriptomics_data - self.cell_mean)/self.cell_std
            all_lines_dict = {cid:np.array(cell).reshape(1,-1) for cid,cell in zip(self.all_transcriptomics_data.index,self.all_transcriptomics_data.values)}
            self.Xs_pos_name_mapper = {pos:col for pos,col in enumerate(self.all_transcriptomics_data.columns)}
            self.Xs = IndexedArray(all_lines_dict)

        else:
            cell_lines_dict = {cid:np.array(cell).reshape(1,-1) for cid,cell in zip(self.cell_lines_data.index,self.cell_lines_data.values)}
            self.Xs_pos_name_mapper = {pos:col for pos,col in enumerate(self.cell_lines_data.columns)}
            self.Xs = IndexedArray(cell_lines_dict)

        self.x_are_scaled = True
        
        ##Y scaling and formatting##
        #compute mean and std for each drug
        self.drug_mean_dict = self.meta_train[['DrugID','Y']].groupby('DrugID').mean()
        self.drug_mean_dict = pd.Series(data=self.drug_mean_dict['Y'].values,index=self.drug_mean_dict.index).to_dict()
        self.drug_std_dict = self.meta_train[['DrugID','Y']].groupby('DrugID').std()
        self.drug_std_dict = pd.Series(data=self.drug_std_dict['Y'].values,index=self.drug_std_dict.index).to_dict()

    def _revert_scaling(self,use_external=False):

        #check if data is already scaled
        assert self.x_are_scaled, "Data is not scaled"

        if use_external:
            self.cell_lines_data = self.cell_lines_data*self.cell_std + self.cell_mean
            self.all_transcriptomics_data = self.all_transcriptomics_data*self.cell_std + self.cell_mean

        else:
            self.cell_lines_data = self.cell_lines_data*self.cell_std + self.cell_mean


    #define some getter methods
    def get_genes(self):
        return self.genes
    
    def get_drugs_ids(self):
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
    
    def get_drug_mean(self,drugID):
        return self.drug_mean_dict[drugID]
    
    def get_drug_std(self,drugID):
        return self.drug_std_dict[drugID]
    
    def get_indexes_sources(self,indexes):
        return self.source_mapper.set_index('index').loc[indexes]['Source'].values



def prepare_data(drugID, dataset, random_state, 
                 gene_selection_mode, 
                 cv_iterations=3,
                 #return_loader=False,
                 use_external_datasets=False,
                 external_dataset=None,
                 data_path=None,celligner_output_path=None,
                 use_dumped_loaders=False,**kwargs):
    
    if (use_external_datasets) and (use_dumped_loaders):
        load_path = Path(data_path)/'loader_dumps'/f'{dataset}_inference'/f'{external_dataset}.pkl'
        #dataset_dump = dataset + '_inference'
    else:
        load_path = Path(data_path)/'loader_dumps'/dataset/f'{random_state}.pkl'

    if use_dumped_loaders:
        #with open(data_path/'loader_dumps'/dataset/f'{random_state}.pkl','rb') as f:
        with open(load_path,'rb') as f:
            loader = pickle.load(f)

    #load data
    else:
        loader = DatasetLoader(dataset=dataset,
                            data_path=data_path,
                            celligner_output_path=celligner_output_path,
                            use_external_datasets=use_external_datasets,
                            samp_x_tissue=2, random_state=random_state)
    
    
    if use_external_datasets:
        #inference for now is only supported for moa primed models
        assert gene_selection_mode == 'moa_primed', "Inference is only supported for moa primed models when using external datasets"
        #if using external inference, asses that the celligner output path is provided
        assert celligner_output_path is not None, "If using external datasets, celligner_output_path must be provided"
        
    #prepare cross validation data and genes if knowledge primed
    if gene_selection_mode == 'moa_primed':
        
        data = []
        #obtain data for each cross validation iteration
        for i in range(cv_iterations):
            data.append(loader.split_and_scale(drugID=drugID, val_random_state=i,use_external=use_external_datasets))

        #get genes selected for the drug
        genes = loader.get_genes()
        gene_getter = GeneGetter(dataset=dataset, data_path=data_path, available_genes=genes)
        genes = gene_getter.get_genes(drugID)

        #subset data for the selected genes
        cv_data = []
        for idx, d in enumerate(data):
            train_X = d['train_X'][genes]
            test_X = d['test_X'][genes]
            valid_X = d['valid_X'][genes]

            #TODO: probably we should get rid of saving test_X in cv_data
            #if we're doing inference on external datasets we only have train and validation (we use all available data), we rejoin train and test to leverage the same code
            if use_external_datasets:
                cv_data.append({'train_X': pd.concat([train_X,test_X]), 'train_Y': pd.concat([d['train_Y'],d['test_Y']]), 'valid_X': valid_X, 'valid_Y': d['valid_Y']})
            #if no inference on external datasets we have all of the splits
            else:
                #cv_data.append({'train_X': train_X, 'train_Y': d['train_Y'], 'valid_X': valid_X, 'valid_Y': d['valid_Y'], 'test_X': test_X, 'test_Y': d['test_Y']})
                cv_data.append({'train_X': train_X, 'train_Y': d['train_Y'], 'valid_X': valid_X, 'valid_Y': d['valid_Y']})
                
        #return cv_data, genes, loader
        out = {'cv_data': cv_data, 'genes': genes, 'loader': loader}

        if use_external_datasets:
            #out['external_X'] = d['external_X'][genes]
            #conatenate everything for external inference
            #out['external_X'] = pd.concat([d['external_X'][genes],d['test_X'][genes],d['train_X'][genes],d['valid_X'][genes]])
            out['external_X'] =d['external_X'][genes]
            out['external_indexes'] = out['external_X'].index
        else:
            out['test_X'] = d['test_X'][genes]
            out['test_Y'] = d['test_Y']
            out['test_indexes'] = d['test_X'].index

        return out

    #prepare data for all genes
    elif gene_selection_mode == 'all_genes':
        data_dict = loader.split_and_scale(drugID=drugID, val_random_state=0)
        data_dict['loader'] = loader
        data_dict['test_indexes'] = data_dict['test_X'].index
        return data_dict
        
    else:
        raise ValueError("Invalid gene_selection_mode. Must be 'knowledge_primed' or 'all_genes'")