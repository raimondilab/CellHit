import sys
import json
import argparse
import numpy as np
from pathlib import Path
from vllm import LLM, SamplingParams

sys.path.append('./../../')
from CellHit.LLMs import generate_prompt
from CellHit.data import get_prism_lfc_drugs_metadata, get_gdsc_drugs_metadata

#set cuda visible device to 1
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--model_path', type=str, default='~/nous_mixtral/')
    args.add_argument('--data_path', type=str, default='./../../data')
    args.add_argument('--results_path', type=str, default='./../../results/')
    args.add_argument('--prompt_path', type=str, default=None)
    args.add_argument('--dataset', type=str, default='gdsc')
    args.add_argument('--temperature', type=str, default=0.3)
    args.add_argument('--top_p', type=str, default=0.90)
    args.add_argument('--max_tokens', type=str, default=1024)

    #parse arguments
    args = args.parse_args()

    model_path = Path(args.model_path).expanduser()
    data_path = Path(args.data_path)
    results_path = Path(args.results_path) / args.dataset

    if args.prompt_path is not None:
        prompt_path = Path(args.prompt_path)
    else:
        prompt_path = data_path/'prompts'/'drug_description_prompt.txt'

    llm = LLM(
            model=model_path,
            revision="gptq-4bit-32g-actorder_True",
            quantization="gptq",
            dtype='float16',
            gpu_memory_utilization=1,
            enforce_eager=True)
            #**{"attn_implementation":"flash_attention_2"}) Turn on for faster decoding (Ampere and newer GPUs only)

    #initialize sampling parameters
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

    #initialize lists to store prompts and drug ids
    prompts = []
    dids = []

    if args.dataset == 'gdsc':
        metadata = get_gdsc_drugs_metadata(data_path)
    else:
        metadata = get_prism_lfc_drugs_metadata(data_path)

    for drug_id, drug_name, mechanism_of_action, putative_targets in metadata[['DrugID','Drug', 'MOA', 'repurposing_target']].values:
    
        #if mechanism_of_action is nan, replace with "Not known"
        if not isinstance(mechanism_of_action, str):
            mechanism_of_action = "Not annoted"
        if not isinstance(putative_targets, str):
            putative_targets = "Not annoted"

        #prompts.append(generate_prompt(drug_name, mechanism_of_action, putative_targets))
        dids.append(drug_id)
        prompts.append(generate_prompt(prompt_path,**{'drug_name':drug_name, 'mechanism_of_action':mechanism_of_action, 'putative_targets':putative_targets}))

    #generate the refined descriptions
    outputs = llm.generate(prompts, sampling_params)

    #Save the refined descriptions
    for output in outputs:
        idx = int(output.request_id)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        drug_id = dids[idx]

        with open(results_path/'drug_summary'/f'{drug_id}.txt','w') as f:
                f.write(generated_text)