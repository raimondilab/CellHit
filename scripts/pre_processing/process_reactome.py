import sys
sys.path.append('./../../')
from pathlib import Path
from CellHit.data import get_reactome_layers, get_pathways_genes, get_genes_pathways, get_pathways_drugs

if __name__ == '__main__':

    reactome_path = Path('./../../data/reactome/')
    
    pathways = get_reactome_layers(reactome_path,1)
    patways_to_genes = get_pathways_genes(pathways)
    genes_to_pathways = get_genes_pathways(patways_to_genes)
    pathways_to_drugs = get_pathways_drugs(pathways)

    #save everything to csv
    pathways.to_csv(reactome_path/'pathways_l1.csv',index=False)
    patways_to_genes.to_csv(reactome_path/'pathways_to_genes.csv',index=False)
    genes_to_pathways.to_csv(reactome_path/'genes_to_pathways.csv',index=False)
    pathways_to_drugs.to_csv(reactome_path/'pathways_to_drugs.csv',index=False)

