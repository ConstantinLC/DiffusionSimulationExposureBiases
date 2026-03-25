#!/usr/bin/env python
"""
Evaluate the diffusion shortcut baseline (Shehata et al., 2025) on a pretrained
DiffusionModel checkpoint.

Instead of starting from pure noise at step T-1, the shortcut starts from an
intermediate noise level t_start:  x_{t_start} = σ_{t_start} * ε,  then runs
only t_start+1 reverse denoising steps.

The optimal t_start is found by sweeping over all values and measuring 1-step
prediction MSE on the validation set.

Usage:
  python experiments/eval_shortcut_sweep.py \
      --checkpoint_dir ./checkpoints/KolmogorovFlow/DiffusionModel_1 \
      --output results/shortcut_sweep_kolmo.pdf

  # Restrict sweep to every Nth step (faster):
  python experiments/eval_shortcut_sweep.py \
      --checkpoint_dir ./checkpoints/KolmogorovFlow/DiffusionModel_1 \
      --stride 2 \
      --output results/shortcut_sweep_kolmo.pdf
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
}


def build_diffusion_model(model_cfg: dict) -> DiffusionModel:
    return DiffusionModel(
        dimension=model_cfg['dimension'],
        dataSize=model_cfg['dataSize'],
        condChannels=model_cfg['condChannels'],
        dataChannels=model_cfg['dataChannels'],
        diffSchedule=model_cfg['diffSchedule'],
        diffSteps=model_cfg['diffSteps'],
        inferenceSamplingMode=model_cfg['inferenceSamplingMode'],
        inferenceConditioningIntegration=model_cfg['inferenceConditioningIntegration'],
        diffCondIntegration=model_cfg['diffCondIntegration'],
        padding_mode=model_cfg.get('padding_mode', 'circular'),
        architecture=model_cfg.get('architecture', 'ours'),
    )


def evaluate_start_step(model: DiffusionModel, val_loader, device: str,
                        start_step: int, n_samples: int = None) -> float:
    """
    MSE of model(cond, start_step=start_step) vs ground truth.

    n_samples: if set, stop after this many samples (for quick sweeps).
    """
    model.eval()
    mse_list = []
    count = 0
    with torch.no_grad():
        for sample in val_loader:
            data = sample["data"].to(device)
            cond = data[:, 0]
            target = data[:, 1]

            pred = model(cond, start_step=start_step)
            mse_list.append(torch.mean((pred - target) ** 2).item())
            count += data.shape[0]
            if n_samples is not None and count >= n_samples:
                break
    return float(np.mean(mse_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True,
                        help='Directory containing config.json and best_model.pth')
    parser.add_argument('--output', default='results/shortcut_sweep.pdf')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=1,
                        help='Evaluate every Nth start_step (1 = all steps)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Limit validation samples per start_step (None = full val set)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Load model config ──────────────────────────────────────────────────────
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    model = build_diffusion_model(config['model_params']).to(args.device)
    ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    print(f"Loaded model from {ckpt_path}")
    print(f"  timesteps = {model.timesteps}")

    # ── Build val loader ───────────────────────────────────────────────────────
    raw_data = config['data_params']
    data_cfg = DataConfig(**{
        **{k: v for k, v in raw_data.items() if k in _DATA_CONFIG_FIELDS},
        'batch_size':     args.batch_size,
        'val_batch_size': args.batch_size,
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    # ── Baseline: full chain (start_step = T-1) ────────────────────────────────
    T = model.timesteps
    print(f"\nBaseline: full chain (start_step = {T-1}) ...")
    mse_full = evaluate_start_step(model, val_loader, args.device,
                                   start_step=T - 1, n_samples=args.n_samples)
    print(f"  MSE (full, {T} steps) = {mse_full:.4e}")

    # ── Sweep start_step ───────────────────────────────────────────────────────
    steps_to_eval = list(range(0, T, args.stride))
    if (T - 1) not in steps_to_eval:
        steps_to_eval.append(T - 1)
    steps_to_eval = sorted(set(steps_to_eval))

    start_steps = []
    mse_values = []

    print(f"\nSweeping {len(steps_to_eval)} start_step values ...")
    for s in steps_to_eval:
        mse = evaluate_start_step(model, val_loader, args.device,
                                  start_step=s, n_samples=args.n_samples)
        nfe = s + 1
        sigmas = model.sqrtOneMinusAlphasCumprod.ravel().cpu()
        sigma_s = sigmas[s].item()
        print(f"  start_step={s:3d}  NFE={nfe:3d}  σ={sigma_s:.4f}  MSE={mse:.4e}")
        start_steps.append(s)
        mse_values.append(mse)

    start_steps = np.array(start_steps)
    mse_values = np.array(mse_values)

    best_idx = int(np.argmin(mse_values))
    best_step = int(start_steps[best_idx])
    best_mse = float(mse_values[best_idx])
    best_sigma = model.sqrtOneMinusAlphasCumprod.ravel().cpu()[best_step].item()
    print(f"\nBest: start_step={best_step}  NFE={best_step+1}  σ={best_sigma:.4f}  MSE={best_mse:.4e}")

    # ── Save results JSON ──────────────────────────────────────────────────────
    results_path = args.output.replace('.pdf', '_results.json')
    sigmas_all = model.sqrtOneMinusAlphasCumprod.ravel().cpu().tolist()
    results = {
        'start_steps': start_steps.tolist(),
        'mse_values': mse_values.tolist(),
        'sigmas': [sigmas_all[s] for s in start_steps.tolist()],
        'best_start_step': best_step,
        'best_mse': best_mse,
        'best_sigma': best_sigma,
        'mse_full_chain': mse_full,
        'T': T,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    nfe_values = start_steps + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, xs, xlabel in [
        (axes[0], nfe_values,    'Number of function evaluations (NFE)'),
        (axes[1], start_steps,   'start_step index'),
    ]:
        ax.axhline(mse_full, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Full chain  ({mse_full:.2e})')
        ax.plot(xs, mse_values, 'o-', color='steelblue', markersize=4,
                label='Shortcut MSE')
        ax.axvline(xs[best_idx], color='red', linestyle=':', linewidth=1.2,
                   label=f'Best  NFE={best_step+1}  ({best_mse:.2e})')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('1-step MSE', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

    axes[0].set_title('Diffusion shortcut: MSE vs NFE', fontsize=12)
    axes[1].set_title('Diffusion shortcut: MSE vs start_step', fontsize=12)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Plot saved to {args.output}")
    plt.close(fig)


if __name__ == '__main__':
    main()
