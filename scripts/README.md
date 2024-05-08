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

