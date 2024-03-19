from scipy.stats import percentileofscore
import numpy as np

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