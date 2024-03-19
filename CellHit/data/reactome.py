import pandas as pd
import networkx as nx
from pathlib import Path
import requests
from tqdm.auto import tqdm
import pubchempy as pcp
import time
from tqdm import tqdm


from .pubchem import get_pubchem_id

def get_reactome_layers(reactome_data_path, layer_number):

    #if reactome is not a Path object, convert it to one
    if not isinstance(reactome_data_path, Path):
        reactome_data_path = Path(reactome_data_path)

    #assert that the files that we need are present
    assert (reactome_data_path / 'ReactomePathways.txt').exists(), 'ReactomePathways.txt not found'
    assert (reactome_data_path / 'ReactomePathwaysRelation.txt').exists(), 'ReactomePathwaysRelation.txt not found'

    reactome_pathways = pd.read_csv(reactome_data_path / 'ReactomePathways.txt', sep='\t', header=None)
    reactome_pathways.columns = ['PathwayID', 'PathwayName', 'Organism']

    # Keep only human pathways
    human_paths = reactome_pathways[reactome_pathways['Organism'] == 'Homo sapiens']
    human_paths_set = set(human_paths['PathwayID'].values)

    reactome_graph = pd.read_csv(reactome_data_path / 'ReactomePathwaysRelation.txt', sep=';')
    # Keep only the human pathways
    reactome_graph = reactome_graph[reactome_graph['Parent'].isin(human_paths_set) & reactome_graph['Child'].isin(human_paths_set)]

    # Initialize the graph
    G = nx.from_pandas_edgelist(reactome_graph, source='Parent', target='Child', create_using=nx.DiGraph())

    # Topological Sort and Layering
    layers = {}
    for node in nx.topological_sort(G):
        incoming = list(G.predecessors(node))
        if not incoming:
            layers[node] = 0
        else:
            #layers[node] = max(layers[pred] for pred in incoming) + 1
            layers[node] = min(layers[pred] for pred in incoming) + 1

    # Create DataFrame with PathwayID, PathwayName, and Layer
    pathways_with_layers = pd.DataFrame([(pid, pname, layers[pid]) for pid, pname in human_paths[['PathwayID', 'PathwayName']].values if pid in layers], columns=['PathwayID', 'PathwayName', 'Layer'])

    if layer_number is not None:
        return pathways_with_layers[pathways_with_layers['Layer'] == layer_number]
    else:
        return pathways_with_layers
    

def get_pathway_genes(path_id):
    
    response = requests.get(f'https://reactome.org/ContentService/data/participants/{path_id.split("-")[-1]}/referenceEntities').json()
    
    genes = []
    
    for i in response:
        if (i['schemaClass'] == 'ReferenceGeneProduct') and ('name' in i):
            genes.append(i['name'][0])
            
    return genes


def get_pathways_genes(pathways):
    
    pathway_genes = []
    
    for path_id in tqdm(pathways['PathwayID']):
        pathway_genes.append(set(get_pathway_genes(path_id)))

    pathways_with_genes = pathways.copy()
    pathways_with_genes['Genes'] = pathway_genes

    #strip withespaces from PathwayName
    pathways_with_genes['PathwayName'] = pathways_with_genes['PathwayName'].apply(lambda x: x.strip())
    
    return pathways_with_genes

def get_genes_pathways(pathways):

    pathway_with_genes = get_pathways_genes(pathways)
    pathway_with_genes['Genes'] = pathway_with_genes['Genes'].apply(lambda x: list(x))
    pathway_with_genes = pathway_with_genes.explode('Genes')

    #group by gene and collect the pathways (both id and name)
    genes_with_pathways = pathway_with_genes.groupby('Genes').agg({'PathwayID': list, 'PathwayName': list}).reset_index()
    #remove genes with no pathways
    genes_with_pathways = genes_with_pathways[genes_with_pathways['PathwayID'].apply(lambda x: len(x) > 0)]
    #remove duplicates
    genes_with_pathways['PathwayID'] = genes_with_pathways['PathwayID'].apply(lambda x: set(x))
    genes_with_pathways['PathwayName'] = genes_with_pathways['PathwayName'].apply(lambda x: set(x))
    
    return genes_with_pathways

def get_pathway_drugs(path_id,path_name):
    response = requests.get(f'https://reactome.org/ContentService/data/participants/{path_id.split("-")[-1]}/referenceEntities').json()
    
    drugs = []
    
    for result in response:
        if (result["className"] == "ReferenceTherapeutic") and ('name' in result):
            tdf = pd.DataFrame()
            tdf['name'] = [result['name'][0]]
            tdf['alternativeNames'] = [','.join(result['name'])]
            tdf['id'] = [result['identifier']]
            tdf['database'] = [result['databaseName']]
            drugs.append(tdf)
            
    if len(drugs) == 0:
        pass
    else:

        out = pd.concat(drugs)
        out['PathwayID'] = [path_id] * len(out)
        out['PathwayName'] = [path_name] * len(out)

        #sort pathway column in front
        cols = out.columns.tolist()
        cols = cols[-2:] + cols[:-2]

        out = out[cols]

        return out
    
def get_pathways_drugs(pathways,annote_pubchem=True):
    
    pathway_drugs = []
    
    for path_id,path_name in tqdm(zip(pathways['PathwayID'],pathways['PathwayName']),total=len(pathways)):
        pathway_drugs.append(get_pathway_drugs(path_id,path_name))

    pathways_with_drugs = pd.concat(pathway_drugs)

    if annote_pubchem:

        def process_drug(drug):
            if drug['database'] == 'PubChem Compound':
                return drug['id'], 'Compound'
            
            else:
                query = get_pubchem_id(drug['name'])

                #wait a couple of seconds
                time.sleep(0.2)

                if query:
                    return query[0], query[1]
                
        tqdm.pandas()
        pathways_with_drugs[['PubChemID','PubChemType']] = pathways_with_drugs.progress_apply(process_drug,axis=1,result_type='expand')

        
    return pathways_with_drugs