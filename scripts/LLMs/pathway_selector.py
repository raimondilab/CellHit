import json
import gc

import argparse
import guidance
import pickle
import torch
import numpy as np

from guidance import select,gen,models

from pathlib import Path

import sys
sys.path.append('./../../')

from CellHit.LLMs import generate_prompt, dictionary_maker, self_consistency
from CellHit.data import get_prism_lfc_drugs_metadata, get_gdsc_drugs_metadata
from CellHit.data import get_reactome_layers

from sqlalchemy import create_engine
from AsyncDistribJobs.models import Job
from AsyncDistribJobs.operations import configure_database
from AsyncDistribJobs.operations import add_jobs, get_jobs_by_state, job_statistics,fetch_job
from AsyncDistribJobs.operations import process_job

@guidance
def inference(lm,prompt, pathway_number, pathways_list,temperature=0.7):

    break_line = "\n"

    lm += prompt

    choosen_pathways = []
    #controlled generation
    for i in range(pathway_number):
        feasible_pathways = [i for i in pathways_list if i not in set(choosen_pathways)]
        lm += f"\nRationale {i+1}:{gen(f'rationale_{i+1}',stop=break_line,temperature=temperature)}"
        lm += f"\nPathway {i+1}:{select(options=feasible_pathways,name=f'pathway_{i+1}')}\n"
        choosen_pathways.append(lm[f'pathway_{i+1}'])
    return lm


def select_drug(results_path,
                data_path,
                drugID,
                temperature=0.7,
                pathway_number=15,
                self_k=5,
                 **args):

    #take language model and reactome pathways from the main
    global lm
    global reactome_pathways

    #check whether a refined description exists
    if Path(results_path/'drug_summary_refined'/f'{drugID}.txt').exists():
        with open(results_path/'drug_summary_refined'/f'{drugID}.txt','r') as f:
            drug_description = f.read()

    #check whether a description exists
    elif Path(results_path/'drug_summary'/f'{drugID}.txt').exists():
        with open(results_path/'drug_summary'/f'{drugID}.txt','r') as f:
            drug_description = f.read()

    #if no description exists, return
    else:
        return

    dict_list = []

    prompt = generate_prompt(data_path/'prompts'/'mixtral_pathway_selector.txt',**{'drug_description':drug_description,'pathways_list':reactome_pathways,'pathway_number':pathway_number})

    try:
            
        for iter in range(self_k):

            lm = lm + inference(prompt,pathway_number=pathway_number, pathways_list=reactome_pathways,temperature=temperature)
            dict_list.append(dictionary_maker(lm))
            lm.reset()
            gc.collect()
            torch.cuda.empty_cache()

        #dump json
        with open(results_path/'reactome_paths'/f'{drugID}.json','w') as f:
            json.dump(self_consistency(dict_list),f)

    except Exception as e:
        pass


def run_full_asynch_selection(args,select_database_path):

    engine = create_engine(f'sqlite:///{select_database_path}_{args["dataset"]}.db')
    configure_database(engine,reset=args['build_select_db'])

    #if specified, build the search database
    if args['build_select_db']:

        if args['dataset'] == 'gdsc':
            drugs_ids = get_gdsc_drugs_metadata(args['data_path'])['DrugID'].tolist()

        else:
            drugs_ids = get_prism_lfc_drugs_metadata(args['data_path'])['DrugID'].tolist()

        jobs_list = [Job(state='pending',payload={'drugID': int(drugID)},cid=f'{drugID}') for drugID in drugs_ids]
        add_jobs(jobs_list)

    while len(get_jobs_by_state('pending')) > 0:
        process_job(select_drug,**args)

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    
    args.add_argument('--dataset', type=str, default='prism')
    args.add_argument('--prompt_path', type=str, default=None)
    args.add_argument('--model_path', type=str, default='~/mixtral/')
    args.add_argument('--selection_mode', type=str, default='asynch')
    args.add_argument('--drugID', type=str, default='1909')
    args.add_argument('--gpu', type=int, default=2)
    args.add_argument('--self_k', type=int, default=5)
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--pathway_number', type=int, default=15)
    args.add_argument('--results_path', type=str, default='./../../results/')
    args.add_argument('--build_select_db', default=False, action='store_true')
    args.add_argument('--data_path', type=str, default='./../../data/')
    
    #parse arguments
    args = args.parse_args()
    args = vars(args)

    args['results_path'] = Path(args['results_path']) / args['dataset']
    args['data_path'] = Path(args['data_path'])
    args['select_database_path'] = args['results_path'] / 'select_database'

    model_path = Path(args['model_path']).expanduser()
    
    #obtain reactome pathways
    reactome_pathways = get_reactome_layers(args['data_path']/'reactome',layer_number=1)['PathwayName'].tolist()

    #understand the GPU architecture
    gpu_name = torch.cuda.get_device_name(args['gpu'])

    if ('A100' in gpu_name) or ('H100' in gpu_name):
        lm = models.Transformers(str(model_path),**{"device_map":f"cuda:{args['gpu']}","revision":"gptq-4bit-32g-actorder_True","attn_implementation":"flash_attention_2"})
    else:
        lm = models.Transformers(str(model_path),**{"device_map":f"cuda:{args['gpu']}","revision":"gptq-4bit-32g-actorder_True"})

    if args['selection_mode'] == 'asynch':
        run_full_asynch_selection(args,args['select_database_path'])
    elif args['selection_mode'] == 'single_drug':
        select_drug(**args)

    


