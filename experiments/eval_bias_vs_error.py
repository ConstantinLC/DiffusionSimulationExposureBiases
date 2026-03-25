#!/usr/bin/env python
"""
For each checkpoint and each noise level sigma, compute scalar means:
  E_clean(sigma)  = mean over samples of MSE( u_theta(sqrt(alpha)*x + sigma*eps), x )
  B_own(sigma)    = mean over samples of E_own / E_clean

Then plot B_own vs sigma and E_clean vs sigma, one curve per checkpoint.

Usage:
  python experiments/eval_bias_vs_error.py \
      --checkpoint_dir checkpoints/KolmogorovFlow/forecasting/DiffusionModel_inverseCosLog-1.875_20_12 \
      --checkpoint_names epoch_501.pth epoch_1001.pth best_model.pth \
      --output results/bias_vs_error.pdf
"""
import os
import sys
import argparse
import json

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

plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.4,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


@torch.no_grad()
def collect_scalars(model, val_loader, device, n_batches=50, T=10):
    """
    Returns (sigmas, mean_clean, mean_bias) each of length T,
    for T evenly-spaced noise levels (highest sigmas, descending order).
    """
    model.eval()

    sigmas_all = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()
    T_full = len(sigmas_all)

    # T evenly-spaced indices from the highest-sigma end, descending
    indices = torch.linspace(T_full - T, T_full - 1, T).long().flip(0)
    sigmas  = sigmas_all[indices]

    spatial_dims = None
    sum_clean = torch.zeros(T)
    sum_bias  = torch.zeros(T)
    count     = torch.zeros(T)

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        data   = sample['data'].to(device)
        cond   = data[:, 0]
        target = data[:, 1]

        if spatial_dims is None:
            spatial_dims = tuple(range(1, target.ndim))

        _, ests_clean = model(conditioning=cond, data=target,
                              return_x0_estimate=True, input_type='clean')
        _, ests_own   = model(conditioning=cond, data=target,
                              return_x0_estimate=True, input_type='own-pred')

        for local_t, global_t in enumerate(indices.tolist()):
            # ests list is built in reversed(range(T_full)), so index 0 = t=T_full-1
            est_idx   = T_full - 1 - global_t
            est_clean = ests_clean[est_idx]
            est_own   = ests_own[est_idx]

            clean_mse = (est_clean - target).pow(2).mean(dim=spatial_dims)
            own_mse   = (est_own   - target).pow(2).mean(dim=spatial_dims)

            valid = clean_mse > 1e-12
            if valid.sum() == 0:
                continue

            ratio = (own_mse / clean_mse.clamp(min=1e-12))[valid]
            sum_clean[local_t] += clean_mse[valid].sum().cpu()
            sum_bias[local_t]  += ratio.sum().cpu()
            count[local_t]     += valid.sum().cpu()

    mean_clean = (sum_clean / count.clamp(min=1)).numpy()
    mean_bias  = (sum_bias  / count.clamp(min=1)).numpy()

    return sigmas.numpy(), mean_clean, mean_bias


def plot(all_results, output_path):
    """
    all_results: list of (label, sigmas, mean_clean, mean_bias)

    One panel per noise level.  Each panel scatters (mean_E_clean, mean_B^own)
    with one point per checkpoint, connected by a line to show training progression.
    """
    sigmas = all_results[0][1]
    T      = len(sigmas)
    ncols  = min(T, 5)
    nrows  = (T + ncols - 1) // ncols

    ckpt_colors = plt.cm.tab10(np.linspace(0, 0.9, len(all_results)))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)

    for local_t in range(T):
        row, col = divmod(local_t, ncols)
        ax = axes[row][col]

        e_vals = [r[2][local_t] for r in all_results]
        b_vals = [r[3][local_t] for r in all_results]

        # Connect checkpoints with a line (training trajectory)
        ax.plot(e_vals, b_vals, '-', color='gray', lw=1, zorder=1)

        for i, (label, _, _, _) in enumerate(all_results):
            ax.scatter(e_vals[i], b_vals[i], s=60, color=ckpt_colors[i],
                       zorder=2, label=label)

        ax.axhline(1.0, color='gray', lw=1, ls='--')
        ax.set_xscale('log')
        ax.set_xlabel(r'$E_{clean}$ (MSE)')
        ax.set_ylabel(r'$\mathcal{B}^{own}$')
        ax.set_title(rf'$\sigma={sigmas[local_t]:.3f}$', fontsize=12)
        ax.legend(fontsize=8)

    for local_t in range(T, nrows * ncols):
        row, col = divmod(local_t, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(r'$\mathcal{B}^{own}$ vs $E_{clean}$ per noise level (each point = one checkpoint)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved → {output_path}")

    # Summary table
    for label, sigmas_ckpt, mean_clean, mean_bias in all_results:
        print(f"\n[{label}]")
        print("{:<8s}  {:>12s}  {:>10s}".format("sigma", "E_clean", "B^own"))
        for local_t in range(T):
            print("{:<8.4f}  {:>12.2e}  {:>10.4f}".format(
                sigmas_ckpt[local_t], mean_clean[local_t], mean_bias[local_t]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--checkpoint_names', nargs='+', default=['best_model.pth'])
    parser.add_argument('--output', default='results/bias_vs_error.pdf')
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--T', type=int, default=10,
                        help='Number of noise levels to plot')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    data_params  = cfg['data_params']
    model_params = cfg['model_params']

    data_cfg = DataConfig(**{k: v for k, v in data_params.items() if k in _DATA_CONFIG_FIELDS})
    _, val_loader, _ = get_data_loaders(data_cfg)

    all_results = []
    for ckpt_name in args.checkpoint_names:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        print(f"\nLoading {ckpt_path} ...")
        model = DiffusionModel(
            checkpoint=ckpt_path,
            load_betas=True,
            dimension=model_params['dimension'],
            dataSize=model_params['dataSize'],
            condChannels=model_params['condChannels'],
            dataChannels=model_params['dataChannels'],
            diffSchedule=model_params['diffSchedule'],
            diffSteps=model_params['diffSteps'],
            inferenceSamplingMode=model_params['inferenceSamplingMode'],
            inferenceConditioningIntegration=model_params['inferenceConditioningIntegration'],
            diffCondIntegration=model_params['diffCondIntegration'],
            padding_mode=model_params.get('padding_mode', 'circular'),
            architecture=model_params.get('architecture', 'ours'),
        ).to(device)
        model.eval()

        model.compute_schedule_variables(model.sqrtOneMinusAlphasCumprod.ravel()[-20:])

        label = os.path.splitext(ckpt_name)[0]
        sigmas, mean_clean, mean_bias = collect_scalars(
            model, val_loader, device, n_batches=args.n_batches, T=args.T)
        all_results.append((label, sigmas, mean_clean, mean_bias))

        del model

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plot(all_results, args.output)


if __name__ == '__main__':
    main()
