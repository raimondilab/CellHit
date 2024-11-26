from scipy.stats import percentileofscore
import numpy as np
from pandarallel import pandarallel
from multiprocessing import cpu_count
class QuantileScoreComputer(object):
    
    def __init__(self,metadata,
                 cell_col='DepMapID',drug_col='DrugID',score_col='Predictions'):
        
        
        #-- cell distributions --
        temp = metadata[[cell_col,score_col]]
        temp = temp.groupby(cell_col)[score_col].agg(list)
        
        temp_dict = temp.to_dict()
        self.distrib_cells = {key: np.array(temp_dict[key]) for key in temp.index}
        
        #-- drug distributions --
        temp = metadata[[drug_col,score_col]]
        temp = temp.groupby(drug_col)[score_col].agg(list)
        
        temp_dict = temp.to_dict()
        self.distrib_drugs = {key: np.array(temp_dict[key]) for key in temp.index}

        
    def compute_score(self, drug, cell, score):
        
        #efficacy
        efficacy = 1-(percentileofscore(self.distrib_cells[cell], score)*0.01)
        
        #selectivity
        selectivity = 1-(percentileofscore(self.distrib_drugs[drug], score)*0.01)
        
        #score as harmonic mean of efficacy and selectivity
        score = 2*(efficacy*selectivity)/(efficacy+selectivity)
        
        return score
    
    def parallel_compute_score(self,df,
                       drug_col='DrugID',cell_col='DepMapID',score_col='Predictions',score_col_out='QuantileScore',
                       n_jobs=-1):
        
        if n_jobs == -1:
            n_jobs = cpu_count()
        
        pandarallel.initialize(nb_workers=n_jobs)
        df[score_col_out] = df.parallel_apply(lambda x: self.compute_score(drug=x[drug_col],cell=x[cell_col],score=x[score_col]),axis=1)
        return df[score_col_out]

    def compute_drug_score(self,drug,score,return_distrib=False):
        score = 1-(percentileofscore(self.distrib_drugs[drug], score)*0.01)
        if return_distrib:
            return score, self.distrib_drugs[drug]
        else:
            return score
    
    def compute_cell_score(self,cell,score,return_distrib=False):
        score = 1-(percentileofscore(self.distrib_cells[cell], score)*0.01)
        if return_distrib:
            return score, self.distrib_cells[cell]
        else:
            return score

    def add_cells(self,cells,
                  cell_col='DepMapID',score_col='Predictions'):
        
        temp = cells[[cell_col,score_col]]
        temp = temp.groupby(cell_col)[score_col].agg(list)
        
        temp_dict = temp.to_dict()
        self.distrib_cells = {**self.distrib_cells,**temp_dict}

    def save(self, filepath):
        """Save the QuantileScoreComputer object to a file using numpy.
        
        Args:
            filepath (str): Path where to save the object
        """
        save_dict = {
            'distrib_cells': self.distrib_cells,
            'distrib_drugs': self.distrib_drugs
        }
        np.save(filepath, save_dict)
        
    @classmethod
    def load(cls, filepath):
        """Load a QuantileScoreComputer object from a file.
        
        Args:
            filepath (str): Path to the saved object
            
        Returns:
            QuantileScoreComputer: Loaded object
        """
        load_dict = np.load(filepath, allow_pickle=True).item()
        
        obj = cls.__new__(cls)
        obj.distrib_cells = load_dict['distrib_cells']
        obj.distrib_drugs = load_dict['distrib_drugs']
        return obj
        