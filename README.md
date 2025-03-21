
# DenseNet Implementation in PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of DenseNet with various configurations. Includes comprehensive training configurations, model checkpoints, and TensorBoard logging.

## Features

- Multiple DenseNet architectures (k=12/24, L=40/100/250 layers)
- Support for DenseNet-BC (Bottleneck & Compression) variants
- Configurable data augmentation strategies
- Automatic dataset downloading and preprocessing
- Experiment tracking with TensorBoard
- Model checkpointing
- Comprehensive configuration management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FILALIHicham/DenseNet.git
cd DenseNet
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Training
Run all experiments defined in the config file:
```bash
python train/train.py --config configs/config.yaml
```

To run specific experiments, edit the config file to comment out unwanted configurations before running.

### Configuration
The `configs/config.yaml` file contains pre-defined experiments with various configurations. Key parameters:

```yaml
dataset:
  name: CIFAR10/CIFAR100/SVHN
  augment: true/false

model:
  growth_rate: 12/24
  block_layers: [12, 12, 12]  # Number of layers per dense block
  bottleneck: true/false
  compression: 0.5            # Compression factor for transitions
  
training:
  epochs: 300
  learning_rate: 0.1
  milestones: [150, 225]      # LR reduction points
```

### Model Architecture
Key architecture components from `models/densenet.py`:
- **Dense Blocks**: Feature concatenation across layers
- **Bottleneck Layers**: 1×1 convolutions for computational efficiency
- **Transition Layers**: Feature compression and spatial downsampling
- **Global Average Pooling**: Before final classification layer

## Results
Best model checkpoints are saved in:
```
logs/checkpoints/{experiment_name}/best_model.pth
```

## Monitoring
TensorBoard logs are saved in:
```
logs/tensorboard/{experiment_name}/
```

Launch TensorBoard with:
```bash
tensorboard --logdir logs/tensorboard
```

## Directory Structure
```
DenseNet/
├── configs/            # Experiment configurations
├── data/               # Dataset storage and loaders
├── logs/               # Training logs and checkpoints
│   ├── checkpoints/    # Saved models
│   └── tensorboard/    # Training metrics
├── models/             # DenseNet implementation
└── train/              # Training scripts
```

## Reference
```bibtex
@misc{huang2018denselyconnectedconvolutionalnetworks,
      title={Densely Connected Convolutional Networks}, 
      author={Gao Huang and Zhuang Liu and Laurens van der Maaten and Kilian Q. Weinberger},
      year={2018},
      eprint={1608.06993},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1608.06993}, 
}
```