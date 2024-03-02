import sys
import json
import time
import argparse
from pathlib import Path
from tqdm.auto import tqdm

sys.path.append('./../../')
from CellHit.LLMs import fetch_abstracts
from CellHit.data import get_prism_lfc_drugs_metadata, get_gdsc_drugs_metadata

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='./../../data')
    args.add_argument('--dataset', type=str, default=None)
    args.add_argument('--mail', type=str, default=None)
    args.add_argument('--k')

    data_path = Path(args.data_path)

    if args.dataset == 'gdsc':
        metadata = get_gdsc_drugs_metadata(data_path)
    else:
        metadata = get_prism_lfc_drugs_metadata(data_path)

    for row in tqdm(metadata.iterrows(), total=len(metadata)):
        drug_name=row[1]['Drug'].lower()
        drug_id=row[1]['DrugID']
        
        try:

            abstracts = fetch_abstracts(drug_name, mail=args.mail, k=args.k)
            abstracts = [str(a) for a in abstracts]
            abstracts = {'drug_id': drug_id, 'drug_name': drug_name,'abstracts': abstracts}

            #create a json and dump the abstracts
            with open(data_path/'drug_abstracts'/args.dataset/f'{drug_id}.json', 'w') as f:
                json.dump(abstracts, f)

            #sleep for 2 seconds
            time.sleep(0.45)

        except:
            with open(f'failed_abstracts_{args.dataset}.txt', 'a') as f:
                f.write(f'{drug_id},{drug_name}\n')

