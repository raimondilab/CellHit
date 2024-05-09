# Learning and actioning general principles of cancer cell drug sensitivity

![Workflow](learning_workflow.png)

This repository contains the complete codebase required to reproduce the analysis presented in the paper [Learning and actioning general principles of cancer cell drug sensitivity](https://www.biorxiv.org/content/10.1101/2024.03.28.586783v2.article-metrics).

## Repository Structure

The code is organized into four main directories, each tailored to facilitate specific aspects of the analysis:

- [**CellHit:**](https://github.com/mr-fcharles/CellHit/tree/master/CellHit) This is a custom library that encapsulates all the functions used throughout the analysis. The library is designed for reusability in further analyses, making it a versatile tool for similar research.

- [**scripts:**](https://github.com/mr-fcharles/CellHit/tree/master/scripts) This folder contains various Python scripts that manage tasks ranging from data pre-processing to model training. An additional Markdown file is included in this folder, providing detailed descriptions and usage instructions for each script.

- [**AsyncDistribJobs:**](https://github.com/mr-fcharles/CellHit/tree/master/AsyncDistribJobs) This directory houses an auxiliary custom library crafted to efficiently manage asynchronous parallel jobs on HPC environments.

- [**Data:**](https://github.com/mr-fcharles/CellHit/tree/master/data) This directory contains data needed to reproduce all the obtained results

# 1. Development Environment Setup

This guide provides detailed steps to set up the development environment necessary to replicate the results from our research. The setup includes creating a new Python environment, installing general libraries, large language model (LLM) libraries, and compiling XGBoost with GPU support.

**NOTE: overall time requirments to setup the enviroment may greatly vary from system to system

## Creating a New Python Environment

First, create a new environment using Conda and install CUDA toolkit:

```bash
conda create -n CellHit python=3.11
conda install -c "nvidia/label/cuda-11.8" cuda-toolkit
```

## General Libraries Installation

Install the following general purpose libraries using pip:

```bash
pip install biopython==1.82 \
			SQLAlchemy==2.0.23 \
			tqdm==4.66.1 \
			torch==2.1.2+cu118 \
			torchaudio==2.1.2+cu118 \
			torchvision==0.16.2+cu118 \
			numba==0.58.1 \
            openpyxl==3.1.2
```

## LLM Libraries Installation

Install libraries specifically used for working with large language models:

```bash
pip install guidance==0.1.8 \
             openai==0.28.1 \
             requests==2.31.0 \
             transformers==2.31.0 \
             auto-gptq==0.6.0+cu118 \
             optimum==1.16.1 \
             peft==0.7.1
```

Installing vLLM with CUDA 11.8 requieres a specific procedure:

```bash
# Install vLLM with CUDA 11.8
export VLLM_VERSION=0.2.7
export PYTHON_VERSION=311
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl

# Re-install PyTorch with CUDA 11.8
pip uninstall torch -y
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118

# Re-install xFormers with CUDA 11.8
pip uninstall xformers -y
pip install --upgrade xformers --index-url https://download.pytorch.org/whl/cu118
```

For additional informations refer to the [original vLLM documentation](https://docs.vllm.ai/en/v0.2.7/)

## Machine Learning Libraries

Install additional machine learning libraries:

```bash
pip install scikit-learn==1.3.2 shap==0.43.0 optuna==3.3.0 
```

## Compiling XGBoost with GPU Support

Follow these steps to compile XGBoost with GPU support, using CUDA 11.8 and gcc 10.2.0:

```bash
# Obtain the code
git clone --recursive https://github.com/dmlc/xgboost

# Compiling XGBoost
cd xgboost
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
make -

# Then install
cd python-package/
pip install .
```

For additional informations refer to the [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/build.html)


Make sure to follow these instructions sequentially to avoid any issues with dependencies and library versions. Once finished reproducing the enviroment, clone the repository with 

```bash
git clone https://github.com/raimondilab/CellHit.git
```

# 2. Getting the data

Most of the data required to replicate the results is in the data folder. However, some files were too large for direct upload to GitHub, particularly 

For **transcriptomics** data for CCLE and TCGA. To access this data, clone the repository and create a `transcriptomics` folder within the `data` folder, alongside `metadata` and `reactome`. Then download:

- [OmicsExpressionProteinCodingGenesTPMLogp1.csv](https://depmap.org/portal/download/all/)
- [TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv](https://xenabrowser.net/datapages/?dataset=TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv&host=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)

If you wish to bypass the celligner steps, a pre-aligned version of these transcriptomics datasets is available [here](https://drive.google.com/file/d/1fJZaoqUvqa93S7SzQ7NNvDkSsnwpffEh/view?usp=sharing).

For **MOA data** you can download `prism_LLM_drugID_to_genes.json` from [here](https://drive.google.com/file/d/1KI4VBgF__txb6LLmeFYGZ6CusuAsqiGl/view?usp=sharing) and place it the `MOA_data` inside the `data` folder


# 3. Getting LLMs weights

For the anlysis we used a [GPTQ quantized version](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ) of Mixtral-8x7B-Instruct-v0.1 from [Mistral.AI](https://mistral.ai). The scripts search for this model in you home folder. To obtain the weights first install `huggingface-cli` and login with the following commands:

```bash
#instal
pip install -U "huggingface_hub[cli]"

#login
huggingface-cli login
```
To login you need a User Access Token from your Settings page (more on this on [huggingface official documentation](https://huggingface.co/docs/hub/security-tokens)). Once logged-in you can obtain the weights of the used LLM with the following command

```bash
huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ --revision gptq-4bit-32g-actorder_True --local-dir <your_home_folder>
```

## Hardware requirements

Our pipeline has primarily been developed and executed in High Performance Computing (HPC) environments. We have utilized the following types of nodes, which are significantly enhanced by GPU acceleration:

- *Daneel* nodes on [HPC@SNS](https://hpccenter.sns.it), each equipped with 2 Intel Xeon CPUs, 36 cores (18 cores per socket), 1.5 TB of RAM (about 42 GB/core), 6TB local scratch space and 4 Tesla (V100) NVIDIA GPUs with 32 GB of RAM (each);
- *Gaia* nodes on [HPC@SNS](https://hpccenter.sns.it), equipped with 2 AMD EPYC 7352, 48 cores (24 physical cores per socket), 512 GB of RAM (10.6 GB/core), 4 NVIDIA A100 and a local scratch area of ~890 GB;
- *BullSequana X2135 "Da Vinci"* nodes on [Leonardo Supercomputer @ CINECA](https://leonardo-supercomputer.cineca.eu), equipped with 1 x CPU Intel Xeon 8358 32 core, 2,6 GHz, 512 (8 x 64) GB RAM DDR4 3200 MHz and 4x Nvidia custom Ampere (A100) GPU 64GB HBM2.

Despite the high-performance hardware requirements for development and training phases, the final trained models are designed to be deployable on standard desktops or laptops without the need for GPU acceleration. These models will be made available upon publication.

## Troubleshooting and Support

If you encounter any issues while setting up or using this environment, please do not hesitate to reach out for help or clarification:

- **Open an Issue:** For problems or enhancements related to the code, please open an issue directly on the [GitHub repository](https://github.com/mr-fcharles/CellHit/issues).

- **Contact via Email:** If you have specific questions or need further assistance, you can email us at `francesco.carli@sns.it`.

We are committed to providing support and making continuous improvements to this project.