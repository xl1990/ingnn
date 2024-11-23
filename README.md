## Information-Enhanced Graph Neural Network for Transcending Homophily Barriers

[*Information-Enhanced Graph Neural Network for Transcending Homophily Barriers*](https://arxiv.org/abs/2210.05382)


Xiao Liu, Lijun Zhang, Hui Guan 

Here are codes to train and evaluate our model INGNN (old name OGNN).

This repo is based on [*Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods*](https://arxiv.org/abs/2110.14446) [*Github*](https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master)

## Organization
`main.py` contains the main full batch experimental scripts.

`main_bilevel_optim.py` contains the main full batch experimental scripts with bi-level optimization.

`parse.py` contains flags for running models with specific settings and hyperparameters. 

`dataset.py` loads datasets.

`models.py` contains implementations for graph machine learning models, though C&S (`correct_smooth.py`, `cs_tune_hparams.py`) are in separate files. Running several of the GNN models on larger datasets may require at least 24GB of VRAM. **Our INGNN (OGNN) model is implemented in this file.**

`homophily.py` contains functions for computing homophily measures, including the one that we introduce in `our_measure`.

`experiments/` contains the bash files to reproduce full batch experiments. 

`results/` contains the csv files of the results for evaluating different models on different datasets.


## Datasets

Download [*data.zip*](https://drive.google.com/file/d/1Z_pdr7q80zMPKDtku5Kc7UKPYRndznhJ/view?usp=sharing) and unzip to the root folder.


## Installation instructions

1. Create a new conda environment using
```
conda env create -f environment.yml -n MY_ENV_NAME
```

2. Activate the environment by
```
conda activate MY_ENV_NAME
```


## Running experiments

1. Make sure a results folder exists in the root directory. 
2. Our experiments are in the `experiments/` directory. 
3. Before running the experiment, you could open `METHOD_exp.sh` and modify `dataset_lst` to change the datasets to run. For example, to run OGNN hyperparameter search: 

```
bash experiments/ognn_exp.sh
```
