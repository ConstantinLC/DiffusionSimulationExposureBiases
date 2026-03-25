#!/usr/bin/env python
"""
Direct experimental validation of Hypothesis 3.1.

Hypothesis 3.1 states that B^(own)(sigma) is:
  (a) an increasing function of E^clean(sigma)
  (b) a decreasing function of sigma

This script loads a trained diffusion model and produces two plots:
  1. B^(own) vs E^clean across noise levels (should be positively correlated)
  2. B^(own) vs sigma (should be decreasing)

Usage:
  python experiments/plot_hypothesis31_direct.py \
      --checkpoint checkpoints/KolmogorovFlow/forecasting/DiffusionModel_psd_100/best_model.pth \
      --config checkpoints/KolmogorovFlow/forecasting/DiffusionModel_psd_100/config.json \
      --output results/hypothesis31_validation.pdf

  # Or use multiple checkpoints from different training stages:
  python experiments/plot_hypothesis31_direct.py \
      --checkpoints epoch50.pth epoch100.pth epoch200.pth \
      --config config.json \
      --output results/hypothesis31_multi_stage.pdf
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from src.utils.diffusion import evaluate_dw_train_inf_gap
from experiments.eval_dw_bias import build_model

# ── ICML style ──────────────────────────────────────────────────────────────
try:
    import seaborn as sns
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
except ImportError:
    pass

rcParams = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times} \usepackage{amsmath} \usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman"],
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.25,
}
mpl.rcParams.update(rcParams)

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
}


@torch.no_grad()
def compute_per_sigma_metrics(model, val_loader, device, n_batches=20):
    """
    Returns per-noise-level E_clean, E_own, B^(own), and sigma values.
    """
    model.eval()
    evals = evaluate_dw_train_inf_gap(
        {'model': model},
        val_loader,
        n_batches=n_batches,
        metric='mse',
        device=device,
        input_types=['clean', 'own-pred'],
    )

    mse_clean = evals['mse_clean']['model'].numpy()
    mse_own = evals['mse_own_pred']['model'].numpy()

    # Get sigmas from model
    if hasattr(model, 'sqrtOneMinusAlphasCumprod'):
        sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu().numpy()
    else:
        sigmas = np.linspace(0, 1, len(mse_clean))

    # B^(own) = E_own / E_clean
    with np.errstate(divide='ignore', invalid='ignore'):
        b_own = np.where(mse_clean > 0, mse_own / mse_clean, np.nan)

    return {
        'sigmas': sigmas,
        'e_clean': mse_clean,
        'e_own': mse_own,
        'b_own': b_own,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config.json')
    parser.add_argument('--output', default='results/hypothesis31_validation.pdf')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_batches', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Load config and build model
    with open(args.config) as f:
        config = json.load(f)

    model = build_model(config['model_params']).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    # Build data loader
    raw_data = config['data_params']
    data_cfg = DataConfig(**{
        k: v for k, v in raw_data.items() if k in _DATA_CONFIG_FIELDS
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    # Compute metrics
    metrics = compute_per_sigma_metrics(model, val_loader, args.device, args.n_batches)

    sigmas = metrics['sigmas']
    e_clean = metrics['e_clean']
    b_own = metrics['b_own']

    # Filter out NaN values
    valid = ~np.isnan(b_own) & (e_clean > 0)
    sigmas_v = sigmas[valid]
    e_clean_v = e_clean[valid]
    b_own_v = b_own[valid]

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): B^(own) vs E^clean  (Hypothesis 3.1a: positive correlation)
    ax = axes[0]
    scatter = ax.scatter(e_clean_v, b_own_v, c=np.log10(sigmas_v), cmap='viridis',
                         s=40, edgecolors='k', linewidths=0.3, zorder=10)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=r'$\mathcal{B}^{(\mathrm{own})} = 1$ (no bias)')

    # Fit and plot trend line
    log_e = np.log10(e_clean_v)
    coeffs = np.polyfit(log_e, b_own_v, 1)
    x_fit = np.linspace(log_e.min(), log_e.max(), 100)
    ax.plot(10**x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=1.5, alpha=0.7,
            label=rf'Linear fit (slope $= {coeffs[0]:.2f}$)')

    # Compute Spearman correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(e_clean_v, b_own_v)
    ax.text(0.05, 0.95, rf'$\rho_s = {rho:.3f}$' + f'\n$p = {pval:.1e}$',
            transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    cb = plt.colorbar(scatter, ax=ax, label=r'$\log_{10}(\sigma)$')
    ax.set_xscale('log')
    ax.set_xlabel(r'Clean-Input Error $\mathcal{E}_{\mathrm{clean}}(\sigma)$')
    ax.set_ylabel(r'Own-Prediction Bias $\mathcal{B}^{(\mathrm{own})}(\sigma)$')
    ax.set_title(r'\textbf{(a)} $\mathcal{B}^{(\mathrm{own})}$ increases with $\mathcal{E}_{\mathrm{clean}}$')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='0.85')

    # Panel (b): B^(own) vs sigma  (Hypothesis 3.1b: decreasing in sigma)
    ax = axes[1]
    ax.scatter(sigmas_v, b_own_v, c=np.log10(e_clean_v), cmap='magma',
               s=40, edgecolors='k', linewidths=0.3, zorder=10)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    cb2 = plt.colorbar(
        ax.collections[0], ax=ax,
        label=r'$\log_{10}(\mathcal{E}_{\mathrm{clean}})$'
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'Noise Level $\sigma$')
    ax.set_ylabel(r'Own-Prediction Bias $\mathcal{B}^{(\mathrm{own})}(\sigma)$')
    ax.set_title(r'\textbf{(b)} $\mathcal{B}^{(\mathrm{own})}$ decreases with $\sigma$')
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='0.85')

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Saved to {args.output}")
    plt.close(fig)

    # Print summary statistics
    print(f"\nHypothesis 3.1 validation summary:")
    print(f"  Spearman corr(E_clean, B_own) = {rho:.4f}  (p = {pval:.2e})")
    print(f"  B^(own) range: [{b_own_v.min():.4f}, {b_own_v.max():.4f}]")
    print(f"  Sigma range: [{sigmas_v.min():.4e}, {sigmas_v.max():.4e}]")
    print(f"  Noise levels with B^(own) > 1: {np.sum(b_own_v > 1)}/{len(b_own_v)}")


if __name__ == '__main__':
    main()
