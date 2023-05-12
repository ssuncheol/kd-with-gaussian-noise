## KD-with-gaussian-noise

This is a Pytorch implementation of Knowledge-distillation with gaussian noise. (Data Parallel)

- Title : Noise as a Resource for Learning in Knowledge Distillation

- Link : [https://arxiv.org/abs/1910.05057]

## Requirements 

```shell
Cuda 11.0
Python3 3.8
PyTorch 1.8 
Torchvision 0.10.0
Einops 0.4.1
```

##  Quickstart 

### Comet(Visualization tool)

- Before starting, you should login wandb using your personal API key. 
- Comet ML : https://www.comet.com/site/

```shell
from comet_ml import Experiment
experiment = Experiment(api_key="YOUR_API_KEY")
```

### Cloning a repository

```shell
git clone https://github.com/ssuncheol/KD-with-gaussian-noise.git
```
