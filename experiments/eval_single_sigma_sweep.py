#!/usr/bin/env python
"""
Collect results from a single-sigma sweep and plot clean vs own-pred MSE
as a function of noise level (sigma).

Expected checkpoint layout:
  <checkpoint_dir>/
    sigma_-2.5/run_1/config.json + best_model.pth
    sigma_-2.4/run_1/config.json + best_model.pth
    ...

Usage:
  python experiments/eval_single_sigma_sweep.py \
      --checkpoint_dir ./checkpoints/KuramotoSivashinsky/single_sigma \
      --output results/single_sigma_sweep.pdf
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
from experiments.eval_dw_bias import build_model, evaluate_dw_bias

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
}


def find_run_dir(sigma_dir: str) -> str | None:
    """Return the latest run_* subdirectory that has best_model.pth."""
    run = [d for d in os.listdir(sigma_dir) if d.startswith('PDERefiner')][0]

    candidate = os.path.join(sigma_dir, run)
    if os.path.exists(os.path.join(candidate, 'best_model.pth')):
        return candidate

    return None


def collect_sigma_dirs(checkpoint_dir: str) -> list[tuple[float, str]]:
    """
    Return a list of (log_sigma, run_dir) sorted by log_sigma,
    one entry per sigma_* directory that has a finished run.
    """
    entries = []
    for name in os.listdir(checkpoint_dir):
        if not name.startswith('sigma_'):
            continue
        log_sigma = float(name[len('sigma_'):])
        sigma_dir = os.path.join(checkpoint_dir, name)
        run_dir = find_run_dir(sigma_dir)
        if run_dir is None:
            print(f"  [skip] {name}: no completed run found")
            continue
        entries.append((log_sigma, run_dir))
    return sorted(entries, key=lambda x: x[0])


@torch.no_grad()
def evaluate_own_pred_k(model, val_loader, device, K):
    """Evaluate own-pred iterated K times; returns per-timestep MSE list."""
    model.eval()
    sample = next(iter(val_loader))
    data = sample["data"].to(device)
    conditioning_frame = data[:, 0]
    target_frame = data[:, 1]

    _, x0_estimates = model(conditioning=conditioning_frame, data=target_frame,
                            return_x0_estimate=True, input_type="own-pred",
                            own_pred_iters=K)

    return [
        torch.mean((x0_estimates[t] - target_frame) ** 2).item()
        for t in range(len(x0_estimates))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir',
                        default='./checkpoints/KuramotoSivashinsky/single_sigma',
                        help='Root directory of the sigma sweep')
    parser.add_argument('--output', default='results/single_sigma_sweep.pdf')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--K', type=int, default=5,
                        help='Number of own-pred iterations for the K-times variant')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Discover runs ─────────────────────────────────────────────────────────
    entries = collect_sigma_dirs(args.checkpoint_dir)
    if not entries:
        print("No completed runs found. Exiting.")
        return
    print(f"Found {len(entries)} sigma runs: {[e[0] for e in entries]}")

    # ── Build shared val_loader from the first config ─────────────────────────
    first_config_path = os.path.join(entries[0][1], 'config.json')
    with open(first_config_path) as f:
        first_config = json.load(f)

    raw_data = first_config['data_params']
    data_cfg = DataConfig(**{
        **{k: v for k, v in raw_data.items() if k in _DATA_CONFIG_FIELDS},
        'batch_size':     1024,
        'val_batch_size': 1024,
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    # ── Evaluate each model ───────────────────────────────────────────────────
    log_sigmas = []
    mse_clean_vals = []
    mse_clean_0_vals = []
    mse_own_pred_vals = []
    mse_own_pred_k_vals = []
    mse_ancestor_vals = []

    for log_sigma, run_dir in entries:
        print(f"\n── sigma = 10^{log_sigma:.1f}  ({run_dir}) ──")

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path) as f:
            config = json.load(f)

        model = build_model(config['model_params']).to(args.device)
        ckpt_path = os.path.join(run_dir, 'best_model.pth')
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        results = evaluate_dw_bias(model, val_loader, args.device)
        mse_own_pred_k = evaluate_own_pred_k(model, val_loader, args.device, K=args.K)

        mse_clean_0    = results['mse_clean'][0]
        mse_clean    = results['mse_clean'][-1]
        mse_own_pred = results['mse_clean_own_pred'][-1]
        mse_k        = mse_own_pred_k[-1]
        mse_ancestor = results['mse_ancestor'][-1]

        print(f"  MSE clean:              {mse_clean:.4e}")
        print(f"  MSE own-pred (1x):      {mse_own_pred:.4e}")
        print(f"  MSE own-pred ({args.K}x):      {mse_k:.4e}")

        log_sigmas.append(log_sigma)
        mse_clean_vals.append(mse_clean)
        mse_clean_0_vals.append(mse_clean_0)
        mse_own_pred_vals.append(mse_own_pred)
        mse_own_pred_k_vals.append(mse_k)
        mse_ancestor_vals.append(mse_ancestor)

    # ── Plot ──────────────────────────────────────────────────────────────────
    log_sigmas_arr = np.array(log_sigmas)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(log_sigmas_arr, mse_clean_vals,       'o-', color='blue',
            label='Clean input (t=0)')
    ax.plot(log_sigmas_arr, mse_clean_0_vals,       'o-', color='purple',
            label='Clean input (t=1)')
    ax.plot(log_sigmas_arr, mse_own_pred_vals,    's-', color='green',
            label='Own-pred 1x (inference)')
    ax.plot(log_sigmas_arr, mse_own_pred_k_vals,  '^-', color='orange',
            label=f'Own-pred {args.K}x (inference)')
    ax.plot(log_sigmas_arr, mse_ancestor_vals,  '^-', color='red',
            label=f'Ancestor-pred (inference)')

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(\sigma)$', fontsize=12)
    ax.set_ylabel('MSE vs ground truth', fontsize=12)
    ax.set_title('Train/inference gap across single-sigma models')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"\nPlot saved to {args.output}")
    plt.close(fig)


if __name__ == '__main__':
    main()
