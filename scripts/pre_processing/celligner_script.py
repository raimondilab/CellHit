import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
from celligner import Celligner


def read_GBM(data_path):

    gbm = pd.read_csv(data_path/'GBM.txt', sep='\t').transpose()
    gbm.columns = [i.split('_')[1] for i in gbm.columns]
    gbm = gbm.loc[:, gbm.std() != 0]

    #take the average of duplicate columns
    gbm = gbm.groupby(gbm.columns, axis=1).mean()

    gbm = gbm.reset_index()
    gbm['index'] = gbm['index'].apply(lambda x: 'FPS_' + x)
    gbm = gbm.set_index('index')

    #add one to all values and than take log2
    gbm = gbm.apply(lambda x: np.log2(x+1))

    return gbm

def read_PDAC():

    hgncID_to_symbol = pd.read_csv(data_path/'hgncID_to_symbol.tsv', sep='\t')
    hgncID_to_symbol = pd.Series(hgncID_to_symbol['Approved symbol'].values, index=hgncID_to_symbol['HGNC ID']).to_dict()

    patiens_metadata = pd.read_csv(data_path/'LMD_RNAseq_annotation.txt', sep='\t')
    patients = set(patiens_metadata['Sample'].values)

    pdac = pd.read_csv(data_path/'LMD_RNAseq_raw.read.counts.txt', sep='\t')
    pdac = pdac[pdac['type_of_gene']=='protein-coding']
    pdac['Gene'] = pdac['HGNC'].map(hgncID_to_symbol)
    pdac = pdac.dropna(subset=['Gene'])
    pdac = pdac[['Gene']+list(patients)]
    pdac = pdac.set_index('Gene')
    pdac = pdac.transpose()

    #remove columns with 0 std
    pdac = pdac.loc[:, pdac.std() != 0]

    #take the average of duplicate columns
    pdac = pdac.groupby(pdac.columns, axis=1).mean()

    pdac = pdac.reset_index()

    pdac['index'] = pdac['index'].apply(lambda x: 'IEO_' + x)
    pdac = pdac.set_index('index')

    #add one to all values and than take log2
    pdac = pdac.apply(lambda x: np.log2(x+1))

    return pdac

def read_external(dataset,data_path):

    if dataset == 'GBM':
        return read_GBM(data_path)
    
    elif dataset == 'PDAC':
        return read_PDAC(data_path)
    
    else:
        raise ValueError('Dataset not found')

    

def source_mapper(x,external=None):
        if x in set(ccle.index):
            return 'CCLE'
        elif x in set(tcga.index):
            return 'TCGA'
        else:
            return external


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--use_external', action='store_true', help='Use external datasets')
    parser.add_argument('--external_dataset', type=str, help='Name of the external dataset')

    args = parser.parse_args()

    ###--CCLE--###
    data_path = Path('../../data/transcriptomics')

    #Read the CCLE data
    ccle = pd.read_csv(data_path/'OmicsExpressionProteinCodingGenesTPMLogp1.csv',index_col=0)
    ccle.columns = [str(i).split(' ')[0] for i in ccle.columns]

    #compute the std of the data
    ccle_stds = ccle.apply(np.std,axis=0)
    #identify the genes with no variance and remove them
    ccle_stds = ccle_stds[ccle_stds>0]
    ccle_stds = set(ccle_stds.index)
    ccle = ccle[[i for i in ccle.columns if i in ccle_stds]]

    ###--TCGA--###
    tcga = pd.read_csv(data_path/'TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv',sep='\t')
    tcga = tcga.set_index('Gene').transpose()

    #remove 0 std columns
    tcga_stds = tcga.apply(np.std,axis=0)
    tcga_stds = tcga_stds[tcga_stds>0]
    tcga_stds = set(tcga_stds.index)
    tcga = tcga[[i for i in tcga.columns if i in tcga_stds]]

    #common columns
    if args.use_external:
        external = read_external(args.external_dataset,data_path)
        common_columns = set(ccle.columns).intersection(tcga.columns).intersection(external.columns)
        ccle = ccle[list(common_columns)]
        tcga = tcga[list(common_columns)]
        external = external[list(common_columns)]

        tumor_samples = pd.concat([tcga,external],axis=0)

    else:
        common_columns = set(ccle.columns).intersection(tcga.columns)
        ccle = ccle[list(common_columns)]
        tcga = tcga[list(common_columns)]

        tumor_samples = tcga

    #align the data
    my_alligner = Celligner()
    my_alligner.fit(ccle)
    my_alligner.transform(tumor_samples)

    #save the data
    output = my_alligner.combined_output.copy()
    output['Source'] = output.index.map(lambda x: source_mapper(x,external=args.external_dataset))
    output = output[list(output.columns[0:1]) + list(output.columns[-1:]) + list(output.columns[:-1])]

    if args.use_external:
        suffix = 'CCLE_TCGA_' + args.external_dataset
    else:
        suffix = 'CCLE_TCGA'

    #save feather file and base alligner
    output.to_feather(data_path/'celligner_{suffix}.feather')
    my_alligner.save(data_path/'base_alligner_{suffix}.pkl')