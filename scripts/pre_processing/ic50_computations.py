import sys
sys.path.append('./../../')

import pandas as pd

from pathlib import Path

from CellHit.utils import ic50_computer

if __name__ == '__main__':

    #set the data path
    data_path = Path('./../../data/experiments/')

    #read the data 
    data = pd.read_csv(data_path/'GBM_experiments.csv')

    #get summary across replicates
    summary = data.groupby(['Sample','Drug','uM']).mean().reset_index()

    samples = []
    drugs = []
    ic50s = []

    for sample in summary['Sample'].unique():
        for drug in summary['Drug'].unique():
            
            tdf = summary[(summary['Sample'] == sample) & (summary['Drug'] == drug)].sort_values('uM',ascending=True)

            ic50 = ic50_computer(tdf['uM'].values,tdf['Viability'].values/100)

            samples.append(sample)
            drugs.append(drug)
            ic50s.append(ic50)
    
    ic50s = pd.DataFrame({'Sample':samples,'Drug':drugs,'IC50':ic50s})

    ic50s.to_csv(data_path/'GBM_ic50s.csv',index=False)

