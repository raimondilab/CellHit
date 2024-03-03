import pandas as pd
import numpy as np
from tqdm import tqdm
import json

from pathlib import Path

import sys
sys.path.append('./../../')

from CellHit.data import get_gdsc_drugs_metadata

import openai
openai.api_key = 'francis' #YOUR API KEY


def GPTQuery(behavior,prompt):

    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0.2,
        messages=[{"role": "system", "content": behavior},
                    {"role": "user", "content": prompt}],
        functions=[{
            "name": "store_drug_info",
            "description": "Stores information about a drug",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "The name of the drug",
                    },
                    "targets": {
                        "type": "array", 
                        "items": {
                                "type": "string",
                                "description": "Gene name in HGNC format",
                            },
                        "description": "The targets of the drug in HGNC format (possibly different from the prompt)",
                    },
                    "short_meachanism_of_action": {
                        "type": "string",
                        "description": "Synthetic description of the mechanism of action given in the prompt",
                    },
                    "summary_mechnism_of_action": {
                        "type": "string",
                        "description": "Summary of the mechanism of action of the drug",
                        },
                    "summary_metabolism": {
                        "type": "string",
                        "description": "Summary of the metabolism of the drug",
                        },
                    "references":{
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Reference in the literature to the information",
                        },
                        "description": "References to the information used to generate the summary",
                        }   
                    }
                }
        }],
        function_call={"name": "store_drug_info"},
        )

    return completion.choices[0].message#.to_dict()

if __name__ == '__main__':

    results_path = Path('./../../results/') #here your path
    
    df = get_gdsc_drugs_metadata(Path('./../../data'))
    
    #df = pd.read_csv('/home/fcarli/DrugByDrug/data/export.csv')
    df = df.replace('-', None)
    #filter all rows that have both pathway_name and targets None
    df = df[df['MOA'].notna() | df['PUTATIVE_TARGET'].notna()]

    #filter all rows that have pathway_name Unclassified and targets None
    df = df[~((df['MOA'] == 'Unclassified') & (df['PUTATIVE_TARGET'].isna()))]

    #read the prompt from an outside file
    with open('./../../data/prompts/GPT4_description_prompt.txt', 'r') as f:
        behavior = f.read()


    for i in tqdm(range(df.shape[0])):

        drug_name = df.iloc[i]["Drug"]

        prompt = f'Drug: {df.iloc[i]["Drug"]}; Drug synonims: {df.iloc[i]["DRUG_SYNONYMS"]}; Target: {df.iloc[i]["PUTATIVE_TARGET"]}; MOA: {df.iloc[i]["MOA"]}'
        #remove every 'None' from the prompt
        prompt = prompt.replace('None', '')
        prompt = prompt.replace('Other', '')

        try:
            #get response from GPT-4
            response = GPTQuery(behavior,prompt)
        except:
            #append drug_name to failed_queries
            with open(results_path / 'failed_queries.txt', 'a') as f:
                f.write(drug_name + '\t' + 'Failed to get response' + '\n')

        try:
            #convert response to dict
            output = json.loads(response.to_dict()['function_call']['arguments'])
        
            #save response as a json file
            with open(results_path /'drug_summary/{drug_name}.json', 'w') as f:
                json.dump(output, f)

        except:
            
            #append drug_name to failed_queries
            with open(results_path/'failed_queries.txt', 'a') as f:
                f.write(drug_name + '\t' + 'Failed to save response in appropriate format' + '\n')

            #dump raw response in a txt file
            with open(results_path / 'failed_queries/{drug_name}.txt', 'w') as f:
                f.write(str(response))
