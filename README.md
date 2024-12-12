# HyperDefender
Welcome to the official repository for our paper **"HyperDefender: A Hybrid Framework Safeguarding GNNs for Hyperbolic Space"** accepted for publication at AAAI 2025!. This repository includes the code and data used in our experiments.

# Paper
Thank you for visiting! You can access the preprint of our paper here.

# Highlights
Our framework is designed to achieve the following:

- Evaluating the vulnerability of GNNs and Hy-GNNs to poisoning and evasion attacks, showing that neither standalone Hy-GNNs nor Euclidean defenses can effectively counter adversarial challenges in hierarchical datasets.

- Analyzing Gromov δ-hyperbolicity to understand how adversarial attacks disrupt the hierarchy in graph networks.

- Presenting HyperDefender, a flexible framework that combines Euclidean defenses to develop robust hyperbolic methods, restoring hierarchical integrity and resisting adversarial attacks.

- Benchmarking HyperDefender's performance under adversarial conditions in node classification tasks, demonstrating its superiority over existing Euclidean defense models.


## Installation

### Step 1: Unzip the Attacks Folder

Unzip the `attacks.zip` folder included in the repository.

### Step 2: Set Up the Environment

1. Create a fresh Conda environment:
   ```bash
   conda create -n hyp_def python=3.8.10
   conda activate hyp_def
   ```

2. Install the required versions of `torch` and `torch_geometric`:
   ```bash
   pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
   ```
   *The above command installs PyTorch with CPU support only.*

3. Install PyTorch Geometric:
   ```bash
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
   ```

4. Install other dependencies:
   ```bash
   pip install tensorly fvcore tabulate numba deeprobust
   ```

---

## Experiment

This repository contains code for training various methods for node classification tasks. Below are the command-line arguments and their descriptions:

### Command-Line Arguments

| Argument            | Description                                                                 | Default       |
|---------------------|-----------------------------------------------------------------------------|---------------|
| `-dataset`          | Dataset identifier: `cora`, `disease`, `airport`                           | (required)    |
| `-he_dim`           | Dimension of hyperbolic embedding                                           | `2`           |
| `-hyla_dim`         | Dimension of HyLa feature                                                  | `100`         |
| `-manifold`         | Hyperbolic space model (`poincare`, etc.)                                  | `poincare`    |
| `-gnn_model`        | GNN model: `SGC`, `GCN`, `sage`, `GAT`, `GNNGuard`                         | `SGC`         |
| `-model`            | Feature model class                                                        | `hyla`        |
| `-attack`           | Attack method: `random`, `mettack`, `rnd`                                  | `None`        |
| `-defence`          | Defence method: `Rwl-GNN`, `Pro-HyLa`                                      | `None`        |
| `-ptb_lvl`          | Perturbation level                                                         | `0.1`         |
| `-lr_e`             | Learning rate for hyperbolic embedding                                     | `0.1`         |
| `-lr_c`             | Learning rate for the GNN module                                           | `0.1`         |
| `-lr_a`             | Learning rate for adjacency matrix update                                  | `0.1`         |
| `-alpha`            | Weight of Frobenius norm                                                   | `1`           |
| `-gamma`            | Weight of GCN                                                              | `1`           |
| `-beta`             | Weight of feature smoothing                                                | `0.1`         |
| `-inner_steps`      | Steps for inner optimization                                               | `2`           |
| `-outer_steps`      | Steps for outer optimization                                               | `1`           |
| `-epochs`           | Number of epochs                                                           | `100`         |
| `-gpu`              | GPU to run on (`-1` for no GPU)                                            | `0`           |
| `-seed`             | Random seed                                                                | `43`          |
| `-use_feats`        | Embed in the feature level (`True`) or node level (`False`)                | `False`       |
| `-tuned`            | Use tuned hyperparameters                                                  | `False`       |
| `-order`            | Order of SGC                                                               | `2`           |
| `-symmetric`        | Use symmetric normalization                                                | `False`       |
| `-original_rwl`     | Use original RWL GNN defense mechanism without HyLa                       | `False`       |
| `-jaccard`          | Use Jaccard pre-processing if provided                                     | `False`       |
| `-original`         | Use vanilla GNN model (GNNGuard, SGC, GCN, sage, GAT) without hyperbolic  | `False`       |
| `-dropout`          | Dropout for GNNGuard                                                       | `0.5`         |

---

### Running the Experiments

1. Modify the `train-ncMain.sh` file in the `nc` folder to reflect your desired configuration.
2. Execute the training script from the `nc` folder:
   ```bash
   bash train-ncMain.sh
   ```

### ProHyLa Experiments

To run ProHyLa experiments on the random-attacked airport dataset, use the provided Python notebook:
- `ProHyla_airport.ipynb`

---

## Notes

For any questions or issues, please feel free to reach out or raise an issue in this repository. Happy experimenting!
