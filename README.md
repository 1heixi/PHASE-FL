# PHASE-FL

**PHASE-FL** (**P**rogressive **H**igh-sparsity **A**ggregation and **S**tructure-stabilized **E**dge **F**ederated **L**earning) is a research codebase for ultra-high-sparsity federated dynamic sparse training.

This repository is built on top of the SparsyFed codebase and is actively adapted for studying federated learning under **extreme sparsity regimes** (e.g., 98% / 99%), with a focus on training dynamics, server-side aggregation stability, and local structural stability.

## Overview

In ultra-high-sparsity federated training, performance is not determined only by the final sparsity target. In practice, three coupled factors matter:

1. **How the model enters the high-sparsity regime**
2. **How prunable parameters are aggregated on the server**
3. **How unstable boundary connections evolve locally during pruning**

PHASE-FL is designed around these three aspects.

### Current framework

The current framework consists of three coordinated components:

- **Dynamic Sparsity Scheduling**  
  Gradually moves the model from a lower initial sparsity to the target ultra-high sparsity, instead of enforcing extreme sparsity from the beginning.

- **Prunable-only Support-aware Aggregation**  
  Applies server-side support-aware soft aggregation only on prunable tensors, while leaving non-prunable tensors unchanged.

- **Boundary-band Hysteresis Pruning**  
  Stabilizes local pruning dynamics near the global Top-K threshold by reducing meaningless flip-flop behavior of boundary connections.

## Project status

This repository is an **active research codebase**, not a polished production framework.

Important notes:

- The code reflects the **current real experimental mainline**, not every previously discussed branch.
- Some ideas explored during development were discarded and are **not part of the final intended method**.
- Formal claims, final numbers, and paper-ready conclusions should be taken from experimental logs and the paper draft, not from this README alone.

## Codebase structure

The most important files in the current mainline are:

```text
project/
├── main.py
├── client/
│   └── client.py
├── dispatch/
│   └── dispatch.py
├── fed/
│   ├── server/
│   │   └── wandb_server.py
│   └── utils/
│       └── support_aware_aggregation_utils.py
├── task/
│   ├── cifar_resnet18/
│   │   ├── dataset.py
│   │   ├── dispatch.py
│   │   ├── models.py
│   │   └── train_test.py
│   └── tiny_imagenet_resnet18/
│       ├── dataset.py
│       ├── dispatch.py
│       ├── models.py
│       └── train_test.py
└── conf/
    ├── dataset/
    ├── fed/
    ├── task/
    └── *.yaml
```

### Key components

- `project/main.py`  
  Main training entry. Builds prunable flags and performs fail-fast checks for support-aware aggregation.

- `project/client/client.py`  
  Default FL client implementation. Handles parameter exchange, learning-rate scheduling, and local train/test calls.

- `project/fed/server/wandb_server.py`  
  Server wrapper with support-aware aggregation logic.

- `project/fed/utils/support_aware_aggregation_utils.py`  
  Utilities for prunable-only support-aware soft aggregation.

- `project/task/cifar_resnet18/train_test.py`  
  Main local training / pruning logic for CIFAR + ResNet18 experiments.

## Current experimental path

The current main experimental path is:

- `task.model_and_data = CIFAR_RN18`
- `task.train_structure = CIFAR_RN18_FIX_PRUNE`

When running formal experiments, do **not** rely on YAML defaults alone. Explicitly override critical settings from the command line, especially:

- `task.model_and_data`
- `task.train_structure`
- `task.fit_config.run_config.ggmp_lambda=0.0`
- `task.fit_config.run_config.fedmcr_beta=0.0`
- `task.fit_config.run_config.initial_sparsity`
- `task.fit_config.run_config.target_sparsity`

## Installation

This project uses **Poetry**.

### 1. Clone the repository

```bash
git clone https://github.com/1heixi/PHASE-FL.git
cd PHASE-FL
```

### 2. Install dependencies

```bash
poetry install
```

### 3. Activate the environment

```bash
poetry shell
```

## Dataset preparation

The repository currently supports CIFAR and Tiny-ImageNet related experiments through configuration files under:

- `project/conf/dataset/`
- `project/conf/task/`
- `project/task/tiny_imagenet_resnet18/`

Please make sure dataset directories and partition paths are correctly configured before running experiments.

## Example run

A typical CIFAR-100 / ResNet-18 / 98% sparsity experiment may look like:

```bash
poetry run python -m project.main --config-name=cifar_resnet18 \
  fed.num_rounds=1000 \
  task.model_and_data=CIFAR_RN18 \
  task.train_structure=CIFAR_RN18_FIX_PRUNE \
  task.fit_config.run_config.initial_sparsity=0.30 \
  task.fit_config.run_config.target_sparsity=0.98 \
  use_wandb=true \
  strategy=fedavg
```

## Logging and outputs

Training outputs are usually written under:

- `outputs/`
- `wandb/`

These directories should generally **not** be committed to Git.

## Recommended Git ignore

At minimum, ignore the following:

```gitignore
__pycache__/
*.pyc
.venv/
venv/
outputs/
multirun/
wandb/
data/
.DS_Store
.vscode/
.idea/
```

## Citation

If you use this repository in academic work, please cite the corresponding PHASE-FL paper once available.

## Acknowledgement

This repository is built on top of the SparsyFed codebase and has been substantially modified for research on ultra-high-sparsity federated dynamic sparse training.
