# Diffusion Multi-Steps

Research codebase for training diffusion models for **multi-step PDE forecasting**. The core contribution is a two-phase algorithm for constructing adaptive noise schedules that remain stable under autoregressive rollout.

## Problem

Diffusion models trained on single-step predictions become unstable when used autoregressively: prediction errors compound across steps. This project identifies a per-noise-level *bias* metric B^(2S) and constructs schedules where each transition satisfies B^(2S) ≤ τ (default τ = 1.05).

## Datasets

- **Kolmogorov Flow** — 2D turbulent flow, 64×64, 819/102 train/val trajectories
- **Kuramoto-Sivashinsky (KS)** — 1D chaotic PDE, resolution 256
- **Transonic Flow** — 2D/3D compressible flow

Data is stored in HDF5 format at `/mnt/SSD2/constantin/sda/data`.

## Method

### Phase 1 — Schedule Exploration (`train_exploration.py`)

Train on a dense log-uniform grid of noise levels. Progressively mark levels as "solved" when their two-step bias drops below τ. Produces per-level checkpoints and an `exploration_state.json`.

### Phase 2 — Greedy Schedule Construction (`train_greedy_schedule.py`)

Starting from the solved levels, greedily select the largest feasible noise level jumps. Produces a compact `schedule.json` for deployment.

## Models

| Model | Description |
|---|---|
| `DiffusionModel` | Conditional DDPM/DDIM, supports multiple noise schedules |
| `PDERefiner` | Cold-start iterative refinement variant |
| `Unet2D` / `Unet1D` | U-Net backbones |
| `DilResNet` | Dilated residual network |
| `FNO` | Fourier Neural Operator |

## Training

Configuration is managed via [Hydra](https://hydra.cc/). Experiment presets live in `configs/experiment/`.

```bash
# Single-step training
python train.py +experiment=kolmo_multisteps

# Phase 1: noise level exploration
python train_exploration.py +experiment=kolmo_exploration

# Phase 2: greedy schedule construction
python train_greedy_schedule.py +experiment=kolmo_exploration \
    training.exploration_run_dir=./checkpoints/KolmogorovFlow/exploration/run_0

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 train.py +experiment=kolmo_multisteps
```

Key config parameters:

```yaml
training:
  tau: 1.05                    # bias threshold for "solved" levels
  n_exploration_levels: 100    # grid density in Phase 1
  epochs_per_pass: 1000
  backgrad: true               # backprop through trajectory steps
data:
  prediction_steps: 1          # >1 for autoregressive training
```

## Evaluation

```bash
python experiments/eval_kolmo.py --checkpoint_dir /path/to/checkpoint
python experiments/eval_ks.py    --checkpoint_dir /path/to/checkpoint
python experiments/eval_multiple_models.py --checkpoint_dirs /path/a /path/b
```

Metrics: MSE, vorticity correlation, power spectral density, two-step bias B^(2S).

## Project Structure

```
├── train.py                        # Main Hydra entry point
├── train_exploration.py            # Phase 1: exploration
├── train_greedy_schedule.py        # Phase 2: schedule construction
├── configs/
│   ├── config.yaml
│   ├── dataset/                    # kolmogorov, ks, transonic
│   ├── model/                      # diffusion, refiner, unet_2d, ...
│   └── experiment/                 # pre-configured experiment presets
├── src/
│   ├── config.py                   # Pydantic config dataclasses
│   ├── models/                     # DiffusionModel, PDERefiner, UNet, FNO, ...
│   ├── data/                       # Dataset loaders (HDF5)
│   ├── training/                   # Training loops
│   └── utils/                      # Noise schedules, DDP helpers
└── experiments/                    # Evaluation and analysis scripts
```

## Requirements

Key dependencies: `torch`, `hydra-core`, `wandb`, `h5py`, `einops`, `numpy`, `scipy`, `neuraloperator`.

A full pinned environment is available in `wandb/run-*/files/requirements.txt` (generated automatically by W&B on each run).

Experiment tracking is done via [Weights & Biases](https://wandb.ai) (entity: `cleclei`).
