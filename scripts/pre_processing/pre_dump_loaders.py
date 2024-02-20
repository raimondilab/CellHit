import sys
sys.path.append('./../../')
import os
import pickle
from pathlib import Path
from CellHit.data import DatasetLoader,prepare_data
from tqdm.auto import tqdm


if __name__ == '__main__':

    celligner_output_path = Path('./../../data/transcriptomics/celligner_CCLE_TCGA.feather')
    data_path = Path('./../../data/')

    for dataset in ['gdsc','prism']:
        
        for random_state in tqdm(range(20)):

            loader = DatasetLoader(dataset=dataset,
                            data_path=data_path,
                            celligner_output_path=celligner_output_path,
                            use_external_datasets=False,
                            samp_x_tissue=2,random_state=random_state)
            

            with open(data_path/'loader_dumps'/dataset/f'{random_state}.pkl','wb') as f:
                pickle.dump(loader,f)

        #prepare data for external datasets
        print(f'Preparing data for external datasets - {dataset}')

        loader = DatasetLoader(use_external_datasets=True,
                        dataset=dataset,
                        data_path=data_path,
                        celligner_output_path='./../../data/transcriptomics/celligner_CCLE_TCGA_GBM.feather',
                        samp_x_tissue=2,random_state=0)
        
        with open(data_path/'loader_dumps'/f'{dataset}_inference/GBM.pkl','wb') as f:
            pickle.dump(loader,f)

        print('--PDAC (gdsc)--')
        loader = DatasetLoader(use_external_datasets=True,
                        dataset=dataset,
                        data_path=data_path,
                        celligner_output_path='./../../data/transcriptomics/celligner_CCLE_TCGA_PDAC.feather',
                        samp_x_tissue=2,random_state=0)
        
        with open(data_path/'loader_dumps'/f'{dataset}_inference/PDAC.pkl','wb') as f:
            pickle.dump(loader,f)