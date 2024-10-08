**Note** : We do not provide an overall launcher for this project as the entire pipeline is computationally intensive and not feasible to run in a single session, especially not on a commercial computer. To enhance reproducibility, we provide detailed instructions on the order of operations and the expected outputs of the various scripts, along with the commands to execute them.

# 1. Pre-processing

Before executing other computations, all scripts within the pre_processing folder must be run. These scripts are independent of each other, so the order of execution within this subset does not matter. Primarily, these scripts generate files that are already available (precomputed) in the `data` folder; thus, this section is mainly for reproducibility purposes and can be optionally skipped. The most crucial scripts in this folder include:

- **process_reactome.py**: This script processes the raw structure of the Reactome database to produce mappings from pathways to genes, genes to pathways, and pathways to drugs, utilizing only Reactome data. Execute the script using the following command:

```bash
python process_reactome.py
```

- **get_pubchem_ids.py**: This script automates the mapping of free-text drug names to PubChem IDs. It requires the path to the data folder and the specific dataset for which the PubChem IDs should be retrieved (either `gdsc`, `prism`, or `all`). To run the script, use:

```bash
# Default (computes for all datasets and uses ./../../data as default path for the data folder)
python get_pubchem_ids.py

# Only gdsc
python get_pubchem_ids.py --dataset gdsc --data_path ./../../data 

# Only prism
python get_pubchem_ids.py --dataset prism --data_path ./../../data 
```

- **celligner_script.py**: This script aligns CCLE with TCGA transcriptomics data, and possibly additional external data, using Celligner. If external datasets need to be aligned (in our case, GBM and PDAC), they should be specified; otherwise, run the script with the following command:

```bash
python celligner_script.py
```

*Note: typical runtime for this scripts is ~30 min using a batch job with 24 CPUSs and and 120 GBs of RAM on a Gaia node*

# 2. LLMs

- **2.1 fetch_abstracts.py**: This script is designed to fetch abstracts related to drug names from scientific databases and save them as JSON files. It requires specifying the path to the data folder, the dataset (either gdsc or prism), and an email address for API access. Additionally, the number of abstracts (k) to retrieve can be specified. Here's how to use the script:

```bash
#Default usage (uses default path and does not specify dataset or email):
python fetch_abstracts.py

#Specifying dataset and path:
python fetch_abstracts.py --dataset gdsc --data_path ./../../data

#Complete specification (includes email and number of abstracts):
python fetch_abstracts.py --dataset prism --data_path ./../../data --mail user@example.com --k 10
```

*Note: typical runtime for this scripts is ~3 hours using a batch job with 24 CPUSs and and 60 GBs of RAM on a Gaia node. Requires internet access*

- **2.2 generate_drug_summaries.py**: This script is designed to generate textaul drug summaries starting from drug metadata and by using a LLM to process input drug data. It specifically configures the language model to run on a specified GPU, handles various paths for data and results, and accepts multiple command-line arguments to customize the process. Running with predefined parameters is recommended

```bash
#Default usage:
python generate_drug_summaries.py
```

*Note: typical runtime for this scripts is ~8 hours using a batch job with 24 CPUSs, 120 GBs of RAM and 1 A100 40GB GPUS on a Gaia node for processing ~6k drugs*

- **2.3 refine_descriptions.py**: Starting from the descriptions generated by the previous code (2.2), this script is designed to generate refined descriptions integrating insights from abstracted fetched in 2.1 . Running with predefined parameters is recommended

```bash
#Default usage:
python refine_descriptions.py
```

*Note: typical runtime for this scripts is ~8 hours using a batch job with 24 CPUSs, 120 GBs of RAM and 1 A100 40GB GPUS on a Gaia node for processing ~6k drugs*

- **2.4 pathway_selector.py**: This script uses a constrained LLM to select, starting from the generated textual description, which are the Reactome pathways most likely to be involved in determining the efficacy of a drug. The scripts allows for varying different parameters such as the number `pathway_number` to select, the number of self-consistency checks `self_k` to perform , the dataset on which the selection should be performed and whether computations should be done for a single drug or for automatically for multiple drugs (possibly coordinating different asynchronous processes). Here's how to use the script:

```bash
#Basic usage - running the procedure on a single drug on prism
python pathway_selector.py --drugID <your_drug_id, ex: 1909> --selection_mode single_drug --dataset prism #prism default dataset

#Basic usage - running the procedure on a single drug on gdsc
python pathway_selector.py --drugID <your_drug_id, ex: 1909> --selection_mode single_drug --dataset gdsc

#Change the number of self-consistency checks
python pathway_selector.py --drugID <your_drug_id, ex: 1909> --selection_mode single_drug --self_k 3 #default is 5

#Run for all the drugs in the dataset (first process that builds the coordination SQAlchemy object)
python pathway_selector.py --build_select_db --dataset prism  --selection_mode asynch

#Run for all the drugs in the dataset (automatically leverages the SQL object created by the command above)
python pathway_selector.py --dataset prism  --selection_mode asynch
```

*Note: typical runtime for this scripts is ~12 minutes using a batch job with 2 CPUSs, 60 GBs of RAM and 1 A100 40GB GPUS and  `self_k` set to 5 on a Gaia node for each drug*

- **2.5 drugs_pathway_annotation.py**: Combines all the info available on a drug to determine the set of genes to use for trainig the MOA-driven model. An important parameter of this script is `self_consistency_threshold` which represents the number of time a pathway needs to be selected in 2.4 in order to be retained (controlling hallucinations). Running with predefined parameters is recommended

```bash
#Default usage:
python drug_pathways_annotation.py

#Changing self consistency threshold
python drug_pathways_annotation.py --self_consistency_threshold 3
```

**NOTE**: scripts 2.2, 2.3 and 2,4 require GPU acceleration, with the first two consuming ~30GB of VRAM and the third one ~38 GB of VRAM

For completness we also report scripts `GPT4_description_prompt.txt` and `GPT4_pathway_prompt.txt` to perform the same procedure leveraging openAI propritary API.


# 3. Search and inference

- **3.1 search_and_inference.py**: This script performs a drug specific hyperparameter optimization using the Optuna library. The scripts allows for varying different parameters such as the number of hyperparameter optimization trials `n_trials`, the `gene_selection_mode` that allows to train all_genes and MOA models from the same script, whether computations should be done for a single drug or for automatically for multiple drugs (possibly coordinating different asynchronous processes). The script also runs inference of the best model on CCLE and TCGA. Here's how to use the script:

```bash
#Basic usage - running the procedure on a single drug on prism
python search_and_inference.py --drugID <your_drug_id, ex: 1909> --selection_mode single_drug --dataset prism #prism default dataset

#Basic usage - running the procedure on a single drug on gdsc
python search_and_inference.py --drugID <your_drug_id, ex: 1909> --selection_mode single_drug --dataset gdsc

#Change the number of optimization trials n_trials
python search_and_inference.py --drugID <your_drug_id, ex: 1909> --selection_mode single_drug --n_trials 100 #default is 300

#Run for all the drugs in the dataset (first process that builds the coordination SQAlchemy object)
python search_and_inference.py --build_search_db --dataset prism  --selection_mode asynch

#Run for all the drugs in the dataset (automatically leverages the SQL object created by the command above)
python search_and_inference.py --dataset prism  --selection_mode asynch
```

*Note: typical runtime for this scripts is ~1 hour using a batch job with 8 CPUSs, 120 GBs of RAM and 1 A100 64GB GPUS on a BullSequana X2135 "Da Vinci" node for processing each drug. Similar performances are obtained on Gaia nodes*

- **3.2 external_inference.py**: This script is used to do training and inference of the best model for each drug on the GBM and PDAC external datasets. Allows to perform computations for a single drug or for automatically for multiple drugs (possibly coordinating different asynchronous processes)

```bash
#Basic usage - running the procedure on a single drug on prism
python external_inference.py --drugID <your_drug_id, ex: 1909> --selection_mode single_drug --dataset prism #prism default dataset

#Basic usage - running the procedure on a single drug on gdsc
python external_inference.py --drugID <your_drug_id, ex: 1909> --selection_mode single_drug --dataset gdsc

#Run for all the drugs in the dataset (first process that builds the coordination SQAlchemy object)
python external_inference.py --build_inference_db --dataset prism  --selection_mode asynch

#Run for all the drugs in the dataset (automatically leverages the SQL object created by the command above)
python external_inference.py --dataset prism  --selection_mode asynch
```

- **3.3 results_completer.py**: This script packs all of the forecasts on external datasets (TCGA, GBM, PDAC) together with record-wise SHAP values and patients metadata into one final big table. Here's how to use the script:

```bash
#Basic usage - inference for PRISM drugs only on TCGA
python results_completer.py  --dataset prism --external_dataset None

#Basic usage - inference for GDSC drugs only on TCGA
python results_completer.py  --dataset gdsc --external_dataset None

#Inference for GDSC drugs only on TCGA and GBM
python results_completer.py  --dataset gdsc --external_dataset GBM

#Inference for GDSC drugs only on TCGA and PDAC
python results_completer.py  --dataset gdsc --external_dataset GBM
```

# 4. Importance

**compute_importances.py**:  This script is used to compute interpretability scores (both SHAP and Permutation Importance) for the trained best models. The scripts allows for varying different parameters such as the number of permutations `n_permutations` used to asses permutation importance. The `importance_mode`specifies whether computations should be done for a single drug or for automatically for multiple drugs (possibly coordinating different asynchronous processes). The script takes advantage of GPU acceleratin to speed-up computations and it may be necessary to tweak `chunk_size` to fit gpu card VRAM size. Here's how to use the script:

```bash
#Basic usage - running the procedure on a single drug on prism
python compute_importance.py --drugID <your_drug_id, ex: 1909> --importance_mode single_drug --dataset prism 

#Basic usage - running the procedure on a single drug on gdsc
python compute_importance.py --drugID <your_drug_id, ex: 1909> --importance_mode single_drug --dataset prism 

#Single drug, changing the number of permutation iterations
python compute_importance.py --drugID <your_drug_id, ex: 1909> --importance_mode single_drug --dataset prism --n_permutations 10

#Run for all the drugs in the dataset (first process that builds the coordination SQAlchemy object)
python compute_importance.py --build_importance_db --dataset prism  --selection_mode asynch

#Run for all the drugs in the dataset (automatically leverages the SQL object created by the command above)
python compute_importance.py --dataset prism  --selection_mode asynch

*Note: typical runtime for this scripts is ~12 minutes using a batch job with 8 CPUSs, 120 GBs of RAM and 1 A100 40GB GPUS and  `n_permutations` set to 3 on a Gaia node for each drug*