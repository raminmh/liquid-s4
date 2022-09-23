# Liquid State Space Models

This repository provides implementations of the Liquid S4 state-space models. The repository is a recent fork of the S4 repo (https://github.com/HazyResearch/state-spaces). It includes the Liquid-S4 KB and PB kernels.

## Setup

### Requirements
This repository requires Python 3.8+ and Pytorch 1.9+.
This repository requires Python 3.8+ and Pytorch 1.9+.
Other packages are listed in `requirements.txt`.


## Run Liquid-S4 models

```bash
# plain S4
python3 -m train wandb=null experiment=lra/s4-lra-imdb

# Old direct polynomial B liquid Kernel:
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=polyb
# Product of Kbar and B liquid kernel:
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=kb

# both kernel with higher degree:
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=polyb model.layer.liquid_degree=3
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=kb model.layer.liquid_degree=3

# Liquid S4D
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=polyb model.layer.mode=diag

# Slightly faster kernel that only uses neighboring terms instead of all combinations
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=polyb model.layer.allcombs=False
```


### Data

#### Datasets and Dataloaders
All logic for creating and loading datasets is in `src/dataloaders`.
This folder may include old and experimental datasets.
The datasets that we consider core are located in `src/dataloaders/datasets.py`.


#### Data
The raw data should be organized as follows.
The data path can be configured by the environment variable `DATA_PATH`, or defaults to `./data` by default, where `.` is the top level directory of this repository (e.g. 'state-spaces').

Most of the dataloaders download their datasets automatically if not found.
External datasets include Long Range Arena (LRA), which can be downloaded from their [GitHub page](https://github.com/google-research/long-range-arena),
and the WikiText-103 language modeling dataset, which can be downloaded by the `getdata.sh` script from the [Transformer-XL codebase](https://github.com/kimiyoung/transformer-xl).
These external datasets should be organized as follows:
```
DATA_PATH/
  pathfinder/
    pathfinder32/
    pathfinder64/
    pathfinder128/
    pathfinder256/
  aan/
  listops/
  wt103/
```
Fine-grained control over the data directory is allowed, e.g. if the LRA ListOps files are located in `/home/lra/listops-1000/`, you can pass in `+dataset.data_dir=/home/lra/listops-1000` on the command line

### Cauchy Kernel using Pykeops

This version is provided by the [pykeops library](https://www.kernel-operations.io/keops/index.html).
Installation usually works out of the box with `pip install pykeops==1.5 cmake` which are provided in the requirements file.

Note that running in a Colab requires installing a different pip package; instructions can be found in the pykeops documentation.

## S4 Experiments

This section describes how to use the latest S4 model and reproduce experiments immediately.
More detailed descriptions of the infrastructure are in the subsequent sections.

### Structured State Space (S4)



### Long Range Arena (LRA)


### Optimizer Hyperparameters

One notable difference in this codebase is that some S4 parameters use different optimizer hyperparameters. In particular, the SSM kernel is particularly sensitive to the A, B, and dt parameters, so the optimizer settings for these parameters are usually fixed to learning rate 0.001 and weight decay 0.

Our logic for setting these parameters can be found in the `OptimModule` class under `src/models/sequence/ss/kernel.py` and the corresponding optimizer hook in `SequenceLightningModule.configure_optimizers` under `train.py`.

## Training

The core training infrastructure of this repository is based on [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) with a configuration scheme based on [Hydra](https://hydra.cc/docs/intro/).
The structure of this integration largely follows the Lightning+Hydra integration template described in https://github.com/ashleve/lightning-hydra-template.

The main experiment entrypoint is `train.py` and configs are found in `configs/`.
In brief, the main config is found at `configs/config.yaml`, which is combined with other sets of configs that can be passed on the command line, to define an overall YAML config.
Most config groups define one single Python object (e.g. a PyTorch nn.Module).
The end-to-end training pipeline can broken down into the following rough groups, where group XX is found under `configs/XX/`:
```
model: the sequence-to-sequence model backbone (e.g. a src.models.sequence.SequenceModel)
dataset: the raw dataset (data/target pairs) (e.g. a pytorch Dataset)
loader: how the data is loaded (e.g. a pytorch DataLoader)
encoder: defines a Module that interfaces between data and model backbone
decoder: defines a Module that interfaces between model backbone and targets
task: specifies loss and metrics
```


### Hydra

It is recommended to read the Hydra documentation to fully understand the configuration framework. For help launching specific experiments, please file an Issue.

### Registries

This codebase uses a modification of the hydra `instantiate` utility that provides shorthand names of different classes, for convenience in configuration and logging.
The mapping from shorthand to full path can be found in `src/utils/registry.py`.

### WandB

Logging with [WandB](https://wandb.ai/site) is built into this repository.
In order to use this, simply set your `WANDB_API_KEY` environment variable, and change the `wandb.project` attribute of `configs/config.yaml` (or pass it on the command line `python -m train .... wandb.project=s4`).

Set `wandb=null` to turn off WandB logging.



## Overall Repository Structure
```
configs/         config files for model, data pipeline, training loop, etc.
data/            default location of raw data
extensions/      CUDA extension for Cauchy kernel
src/             main source code for models, datasets, etc.
  callbacks/     training loop utilities (e.g. checkpointing)
  dataloaders/   data loading logic
  models/        model backbones
    baselines/   misc. baseline models
    functional/  mathematical utilities
    nn/          standalone modules and components
    hippo/       core HiPPO logic
    sequence/    sequence model backbones and layers including RNNs and S4/LSSL
  tasks/         encoder/decoder modules to interface between data and model backbone
  utils/
sashimi/         SaShiMi README and additional code (generation, metrics, MTurk)
train.py         training loop entrypoint
```



## Citation

```
@article{hasani2022liquid,
  title={Liquid Structural State-Space Models},
  author={Hasani, Ramin and Lechner, Mathias and Wang, Tsun-Huang and Chahine, Makram and Amini, Alexander and Rus, Daniela},
  journal={arXiv preprint arXiv:},
  year={2022}
}

```
