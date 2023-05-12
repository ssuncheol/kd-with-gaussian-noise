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

## Experiments 

I use CIFAR10, MNIST Dataset to train and evalute model 


### Arguments
| Args 	| Type 	| Description 	| Default|
|:---------:|:--------:|:----------------------------------------------------:|:-----:|
| Epochs 	| [int] 	| Epochs | 200|
| Data | [str] | Cifar10, mnist | mnist |
| Mode | [str] | train, kd | train | 
| Batch_size 	| [int] 	| Batch size | 128|
| Model 	| [str]	| resnet18, wrn, wrn_s | resnet18 |
| Learning rate | [float] | Learning rate | 1e-1 |
| Weight_decay 	| [float]	| Weight decay | 5e-4 |
|Momentum| [float]| Momentum| 0.9 | 
|Alpha| [float] | Alpha | 0.9 |
| Noise_label | [float] | Noise_label | 0.1 |
| Temp | [float] | Temperature scaling | 10 |


### How to train


```shell
python3 main.py --mode='train' --data='mnist' --batch_size=128 --weight_decay=5e-4 --alpha=0.9 --temp='10' --noise_label=0.1
```
