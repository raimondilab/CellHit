import sys
import json
import argparse
import numpy as np
from pathlib import Path
from vllm import LLM, SamplingParams

sys.path.append('./../../')
from CellHit.LLMs import generate_prompt

#set cuda visible device to 1
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
        prompt_path = data_path/'prompts'/'drug_refiner_prompt.txt'

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

    #obtain the list of drugs for wich the description was generated
    drugs = list((results_path/'drug_summary').iterdir())

    for drug in drugs:

        #obtain the drug id from the path
        drug_id = drug.stem
        dids.append(drug_id)

        #check if abstracts are available, if not, skip
        if Path(data_path/args.dataset/'drug_abstracts'/f'{drug_id}.json').exists():

            #load the pre-fetched abstacts
            with open(data_path/'drug_abstracts'/args.dataset/f'{drug_id}.json') as f:
                abstracts = json.load(f)['abstracts']
            abstracts = "\n".join([f"Abstract {i+1}: {abstract}" for i, abstract in enumerate(abstracts)])

            #load the pre-generated drug description
            with open(results_path/'drug_summary'/f'{drug_id}.txt') as f:
                description = f.read()

            #create a list of prompts
            prompts.append(generate_prompt(prompt_path,**{'drug_description':description,'drug_abstracts':abstracts}))

        else:
            pass

    #generate the refined descriptions
    outputs = llm.generate(prompts, sampling_params)

    #Save the refined descriptions
    for output in outputs:
        idx = int(output.request_id)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        drug_id = dids[idx]

        with open(results_path/'drug_summary_refined'/f'{drug_id}.txt','w') as f:
                f.write(generated_text)