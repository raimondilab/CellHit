# Introduction

This repository contains the complete codebase required to reproduce the analysis presented in the paper [Learning and actioning general principles of cancer cell drug sensitivity](https://www.biorxiv.org/content/10.1101/2024.03.28.586783v2.article-metrics).

## Repository Structure

The code is organized into three main directories, each tailored to facilitate specific aspects of the analysis:

- [**CellHit:**](https://github.com/mr-fcharles/CellHit/tree/master/CellHit) This is a custom library that encapsulates all the functions used throughout the analysis. The library is designed for reusability in further analyses, making it a versatile tool for similar research.

- **scripts:** This folder contains various Python scripts that manage tasks ranging from data pre-processing to model training. An additional Markdown file is included in this folder, providing detailed descriptions and usage instructions for each script.

- **AsyncDistribJobs:** This directory houses an auxiliary custom library crafted to efficiently manage asynchronous parallel jobs on a High-Performance Computing (HPC) environment.

# 1. Development Environment Setup

This guide provides detailed steps to set up the development environment necessary to replicate the results from our research. The setup includes creating a new Python environment, installing general libraries, large language model (LLM) libraries, and compiling XGBoost with GPU support.

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
			numba==0.58.1
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

## Troubleshooting and Support

If you encounter any issues while setting up or using this environment, please do not hesitate to reach out for help or clarification:

- **Open an Issue:** For problems or enhancements related to the code, please open an issue directly on the [GitHub repository](https://github.com/mr-fcharles/CellHit/issues).

- **Contact via Email:** If you have specific questions or need further assistance, you can email us at `francesco.carli@sns.it`.

We are committed to providing support and making continuous improvements to this project.