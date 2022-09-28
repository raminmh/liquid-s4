# Liquid State Space Models ([Paper](https://arxiv.org/abs/2209.12951))

This repository provides the implementation of Liquid S4 state-space models. Liquid-S4 takes state-spaces models to another level by utilizing a linearized version of [liquid neural networks](https://github.com/raminmh/liquid_time_constant_networks) at its core. Read the preprint for more details:
https://arxiv.org/abs/2209.12951

The repository is a recent fork of the S4 repo (https://github.com/HazyResearch/state-spaces). It includes the Liquid-S4 KB and PB kernels.

## Setup

### Requirements
This repository requires Python 3.8+ and Pytorch 1.9+.  
Other packages are listed in `requirements.txt`.

`pip3 install -r requirement.txt`

To install the Custom Cauchy Kernel (more efficient) developed by [Gu et al. 2022](https://github.com/HazyResearch/state-spaces):
```bash
cd extensions/cauchy/
python3 setup.py install
```

You can also skip installing the custom cauchy kernel, and use the kernel provided by the [PyKeOps Library](https://www.kernel-operations.io/keops/python/installation.html). If you `pip3 install -r requirement.txt`, this package will be installed. 

## Datasets:

sCIFAR is downloaded automatically when running a training job.  
Speech Commands dataset is downloaded automatically when running a training job.  
All Long Range Arena (LRA) (except IMDB and Cifar which are auto-downloaded) tasks could be downloaded directly from ths gziped file: [LRA Full Dataset](https://storage.googleapis.com/long-range-arena/lra_release.gz).  
After downloading the LRA task, organize a `data/` folder with the following directory structure:

```bash
$data/
  pathfinder/
    pathfinder32/
    pathfinder64/
    pathfinder128/
    pathfinder256/
  aan/
  listops/
```

If the IMDB dataset did not get downloaded, you can run the following script to download it: [src/dataloaders/imdb_dataset.sh](https://github.com/raminmh/liquid-s4/blob/main/src/dataloaders/imdb_dataset.sh) This bash script places the imdb dataset into a proper 

## Train Liquid-S4 Models

```bash
# plain S4
python3 -m train wandb=null experiment=lra/s4-lra-imdb

# Liquid-S4 PB Kernel: (PB kernels are faster and perform better than KB)
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=polyb
# liquid-S4 KB Kernel:
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=kb

# Increase Liquid Order:
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=polyb model.layer.liquid_degree=3
python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=kb model.layer.liquid_degree=3

```


The default config files are all included in the `config/` folder. To run each experiment change the flag `experiment=` to any of the following YAML files:  

```bash
lra/s4-lra-listops # Long Range Arena: Listops
lra/s4-lra-imdb # Long Range Arena: IMDB Character Level Sentiment Classification (text)
lra/s4-lra-cifar # Long Range Arena: Sequential CIFAR (image)
lra/s4-lra-aan # Long Range Arena: AAN (Retreival)
lra/s4-lra-pathfinder # Long Range Arena: Pathfinder
lra/s4-lra-pathx # Long Range Arena: Path-x

sc/s4-sc  # Speech Commands Recognition Full Dataset

bidmc/s4-bidmc # BIDMC Heart Rate (HR), Raspiratory Rate (RR), and Blood Oxygen Saturation (SpO2)


#Example: 
python3 -m train wandb=null experiment=lra/s4-lra-listops model.layer.liquid_kernel=polyb model.layer.liquid_degree=2

```

### Optimizer Hyperparameters from [S4 Repo](https://github.com/HazyResearch/state-spaces)

One notable difference in this codebase is that some S4 parameters use different optimizer hyperparameters. In particular, the SSM kernel is particularly sensitive to the A, B, and dt parameters, so the optimizer settings for these parameters are usually fixed to learning rate 0.001 and weight decay 0.

Our logic for setting these parameters can be found in the `OptimModule` class under `src/models/sequence/ss/kernel.py` and the corresponding optimizer hook in `SequenceLightningModule.configure_optimizers` under `train.py`.

## Training from [S4 Repo](https://github.com/HazyResearch/state-spaces):

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

### Hydra from [S4 Repo](https://github.com/HazyResearch/state-spaces)

It is recommended to read the Hydra documentation to fully understand the configuration framework. For help launching specific experiments, please file an Issue.

### Registries from [S4 Repo](https://github.com/HazyResearch/state-spaces)

This codebase uses a modification of the hydra `instantiate` utility that provides shorthand names of different classes, for convenience in configuration and logging.
The mapping from shorthand to full path can be found in `src/utils/registry.py`.

### WandB from [S4 Repo](https://github.com/HazyResearch/state-spaces)

Logging with [WandB](https://wandb.ai/site) is built into this repository.
In order to use this, simply set your `WANDB_API_KEY` environment variable, and change the `wandb.project` attribute of `configs/config.yaml` (or pass it on the command line `python -m train .... wandb.project=s4`).

Set `wandb=null` to turn off WandB logging.


## Citation

```
@article{hasani2022liquid,
  title={Liquid Structural State-Space Models},
  author={Hasani, Ramin and Lechner, Mathias and Wang, Tsun-Huang and Chahine, Makram and Amini, Alexander and Rus, Daniela},
  journal={arXiv preprint arXiv:2209.12951},
  year={2022}
}

```
