# Bridging Diffusion and Simulation Exposure Biases for Autoregressive PDE Simulations

Research codebase accompanying the paper **"Bridging Diffusion and Simulation Exposure Biases for Autoregressive PDE Simulations"**.

## Problem

Diffusion models trained on single-step predictions suffer from two compounding issues in autoregressive rollout:
1. **Diffusion exposure bias** — standard noise schedules are sub-optimal for reconstruction; the model is never trained on its own (noisy) outputs.
2. **Simulation exposure bias** — unrolled training to stabilize long-horizon rollouts is prohibitively expensive for diffusion models, since each training step requires a full Markov chain.

## Method

The approach has two stages:

### Stage 1 — Adaptive Schedule Construction

The goal is to find the noise schedule that minimises reconstruction error subject to the constraint that every denoising transition is *stable*, i.e. the two-step bias B^(2S)(σ_t, σ_{t+1}) ≤ τ. A sub-schedule that violates this at any step amplifies errors and is strictly suboptimal.

**Exploration** (`train_exploration.py`): A single training run over a dense log-uniform grid of noise levels. Each level is independently optimised and marked *solved* once its B^(2S) drops below τ (default 1.05). Produces per-level checkpoints and an `exploration_state.json`.

**Greedy construction** (`train_greedy_schedule.py`): Using only forward passes through the solved checkpoints, the optimal schedule is built greedily — starting from the smallest solved σ, each step selects the largest feasible jump that stays within the stability bound. Produces a compact `schedule.json`.

**Final training** (`train.py`): The model is trained from scratch on the constructed schedule with importance sampling weighted by the schedule.

### Stage 2 — Proxy Unrolled Training

With a low-bias model in hand (from Stage 1), a faithful *proxy estimate* of the model's output can be obtained using only the final n denoising steps (n=1 in practice), rather than a full chain. This proxy is used as a fast substitute for full diffusion sampling during unrolled fine-tuning, enabling stable long-horizon rollouts at a fraction of the cost.

## Datasets

- **Kolmogorov Flow** — 2D turbulent flow, 64×64
- **Kuramoto-Sivashinsky (KS)** — 1D chaotic PDE, resolution 256
- **Transonic Flow** — 2D compressible flow around an airfoil

## Models

| Model | Description |
|---|---|
| `DiffusionModel` | Conditional DDPM/DDIM, supports multiple noise schedules |
| `PDERefiner` | Cold-start iterative refinement variant |
| `Unet2D` / `Unet1D` | U-Net backbone with attention |
| `DilResNet` | Dilated residual network |
| `FNO` | Fourier Neural Operator |

## Training

Configuration is managed via [Hydra](https://hydra.cc/). Experiment presets live in `configs/experiment/`.

```bash
# Stage 1a: noise level exploration
python train_exploration.py +experiment=kolmo_exploration

# Stage 1b: greedy schedule construction
python train_greedy_schedule.py +experiment=kolmo_exploration \
    training.exploration_run_dir=./checkpoints/KolmogorovFlow/exploration/run_0

# Stage 1c: final training on constructed schedule
python train.py +experiment=kolmo_multisteps

# Stage 2: proxy unrolled fine-tuning (set backgrad=true, n_proxy_steps>0)
python train.py +experiment=kolmo_multisteps training.backgrad=true training.n_proxy_steps=1

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 train.py +experiment=kolmo_multisteps
```

Key config parameters:

```yaml
training:
  tau: 1.05                    # stability threshold for "solved" levels
  n_exploration_levels: 100    # grid density during exploration
  epochs_per_pass: 1000
  backgrad: true               # enable proxy unrolled training
  n_proxy_steps: 1             # number of denoising steps in proxy estimate
data:
  prediction_steps: 1          # >1 for autoregressive training
```

## Evaluation

```bash
python experiments/eval_kolmo.py --checkpoint_dir /path/to/checkpoint
python experiments/eval_ks.py    --checkpoint_dir /path/to/checkpoint
python experiments/eval_multiple_models.py --checkpoint_dirs /path/a /path/b
```

Metrics: MSE, vorticity correlation, power spectral density, two-step bias B^(2S), high-correlation time.

## Project Structure

```
├── train.py                        # Main Hydra entry point
├── train_exploration.py            # Stage 1: exploration
├── train_greedy_schedule.py        # Stage 1: greedy schedule construction
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
