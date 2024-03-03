import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import os

import sys
sys.path.append('./../../')

from CellHit.data import get_gdsc_drugs_metadata, get_reactome_layers

import openai
# load and set our key
openai.api_key = 'Francis' #YOUR API KEY

def prompt_generator(funcs):

    prompt = ''

    prompt += f'Drug: {funcs["drug_name"]}\n'
    prompt += f'Targets: {",".join(funcs["targets"])}\n'
    prompt += f'MOA: {funcs["summary_mechnism_of_action"]}\n'
    prompt += f'Metabolism: {funcs["summary_metabolism"]}\n'
    prompt += f'References: {",".join(funcs["references"])}\n'

    return prompt

def GPTVariableSelection(behavior, prompt, level1):

    completion_path = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0.2,
        messages=[{"role": "system", "content": behavior},
                {"role": "user", "content": prompt_generator(prompt)}],
        functions=[{
            "name": "store_pathways_info",
            "description": "Stores information about Reactome pathways involved in drug response",
            "parameters": {
                "type": "object",
                "properties": {
                    "pathway_rationale": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Rationale for the choice of the pathway",
                        },
                    },  
                    "pathway_names": {
                        "type": "array", 
                        "items": {
                                "type": "string",
                                "description": "Name of the pathway in Reactome",
                                "enum": level1['PathwayName'].tolist(),
                            },
                        "description": "Name of the feasible Reactome pathways",
                    },
                }
            }
        }],
        function_call={"name": "store_pathways_info"},
    )

    return completion_path.choices[0].message#.to_dict()



if __name__ == '__main__':

    results_path = Path('./../../results/') #here your path

    #reactome layers
    level1 = get_reactome_layers(Path('./../../data/reactome'), layer_number=1)

    with open('./../../data/prompts/GPT4_pathway_prompt.txt','r') as f:
        behavior = f.read()

    #cycle over all the json files in the directory /home/fcarli/DrugByDrug/GPTData/drug_summary
    for file in tqdm(os.listdir(results_path / 'drug_summary')):
        
        #if the file is a json file
        if file.endswith('.json'):
            
            #open the file
            with open(results_path / 'drug_summary' / file,'r') as f:
                
                #load the json
                prompt = json.load(f)
                drug_name = prompt['drug_name']

                try:

                    response = GPTVariableSelection(behavior,prompt)

                    output = json.loads(response.to_dict()['function_call']['arguments'])
                    #save response as a json file
                    with open(results_path / 'drug_variables' / f'{drug_name}.json', 'w') as f:
                        json.dump(output, f)

                except:

                    #append drug_name to failed_variables
                    with open(results_path / 'failed_variables.txt', 'a') as f:
                        f.write(drug_name + '\t' + 'Failed to get response' + '\n')

