<div align="center">
<h1>Weak Learner Debiasing and Masked Debiasing</h1>
</div>

## Introduction

This is the repository for my master's thesis (link coming soon) and for the paper on masked debiasing (link coming soon) under review at EMNLP 2022. It follows the [lightning-hydra template](TODO) to provide an easy-to-use and reproducible experimentation environment. Follow the instructions below to reproduce the results of my thesis and the paper.

All experiments I have carried out for both weak learner debiasing and masked debiasing can be executed with this repository, and a log of all the models can be found in [this public wandb page]().

## Installation

To run this project, you must create a clean python environment and install all the dependencies in `requirements.txt` with `pip`.
Package versions are fixed, and reproduction is not guaranteed if versions are updated.

To allow the scripts to run from the `scripts` folder and successfully import code from `src`, you must add your project base folder to the python path. `export PYTHONPATH="${PYTHONPATH}:/PROJECT/PATH/"`. Most scripts are rough and not meant to be reused. The only important script here is `extract_preds.py` and `extract_squad_preds.py`.

## Basics

The `src` file contains the core code for this project. You will always run `python train.py` or `python scripts/...` instead of accessing `src` code directly. You will find the training pipeline, as well as the necessary Pytorch Lightning modules to make everything work.

The `data` folder contains the necessary source data whenever it is not available in the Huggingface `datasets` package. Each dataloader in `src/datamodules` has code to load and process the raw data. Instructions on how to download and set up each data source is included within the `MEMO.txt` of each dataset folder within `data`.
Credit for each dataset goes to the original authors.

The `scripts` folder contains scripts to perform individual actions, such as evaluating on a specific dataset or extracting predictions. See each file for documentation.

The `config` folder allows you to customize each experiment. Take a look at each yaml file inside `config/experiment` to understand how they are configured. In each case, the file includes all the necessary parameters for reproduction.

I recommend you to create a free Wandb account and configure it in `configs/logger/wandb.yaml`. Then, all runs will be logged through the platform, making experimentation much easier.

## Weak Learner Training

First and foremost, we need the predictions of a weak learner. To train each learner, run the commands below.

Undertrained learner:

```
python train.py experiment=weak_models/undertrain_mnli name=weak/undertrain/mnli seed=10
```

Underparameterized learner:

```
python train.py experiment=weak_models/tinybert_mnli name=weak/tinybert/mnli seed=10
```

Now, you can extract each model's predictions into a prediction file.

```
python scripts/extract_preds.py \\
/PATH/TO/EXPERIMENT/checkpoints/last.ckpt \\
/PATH/TO/EXTRACTION/FOLDER/undertrain-mnli.json \\
--dataset_name mnli
```

You can know your experiment path by looking at the logs, or checking the wandb entry. It is configured to log the experiment path in the `path` parameter.


## Target Model Training

Now you can train a debiased model and a baseline model using the following commands.

Baseline model:

```
python train.py experiment=baselines/bert_mnli name=baseline_mnli seed=10
```

Debiased with a weak model. Simply change the path to debias with different configurations.

```
python train.py experiment=debias/bert_mnli name=debias/[type]/mnli seed=10 datamodule.bias_path=PATH/TO/WEAKMODEL/undertrain-mnli.json
```

## Hydra Tricks

You can modify any parameter from the commandline. For example, do add weight decay, do the following:

```
python train.py experiment=baselines/bert_mnli name=baseline_mnli seed=10 ++model.weight_decay=0.1
```

You can run multiple experiments (for example, try 5 different seeds) by doing a multirun. Here is an example to train the baseline model with 5 seeds:

```
python train.py -m experiment=baselines/bert_mnli name=baseline_mnli seed=10,11,12,13,14 
```

Note that they will run sequentially. To support parallelized multiruns, you must use an advanced launcher / sweeper with hydra, such as ray. I do not cover this here, but you may find some commented-out code in `src/train.py` regarding this.


## Masked Debiasing (Pruning)

In this setting, we load a finetuned baseline model, freeze its weights, and apply movement pruning to perform a mask search. The loss is a debiasing loss coupled with a weak learner, so the mask will be optimized to perform debiasing.

The implementation for this functionality can be found in `src/models/hf_model_pruned.py`. Running it is as easy as the previous experiments:

```
python train.py \\
experiment=pruning/vanilla_then_prune \\
name=pruning/with_undertrained/mnli \\
++datamodule.bias_path="/PATH/TO/WEAKMODEL/undertrain-mnli.json" \\
++model.from_checkpoint="/PATH/TO/EXPERIMENTS/checkpoints/last.ckpt" \\
pruning=unstructured_sigmoid \\
++pruning.final_threshold=0.08 \\
```

In this example, I set `pruning.final_threshold`, which determines the amount of weights that will be pruned. The pruning type is unstructured sigmoid, which means what individual weights are chosen for the mask and a dynamic pruning rate is applied throughout the modules.

## Issues

Feel free to open an issue in the repository if you face any troubles. If I find time, I will try to help.