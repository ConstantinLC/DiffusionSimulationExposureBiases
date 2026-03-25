#!/usr/bin/env python
"""
Test 1: Error spectral centroid shift across training.

For each checkpoint and each noise level sigma, compute the spectral centroid
of the clean-input reconstruction error epsilon_clean:

    k_bar(sigma) = sum_k k * E(k) / sum_k E(k)

where E(k) is the radially-averaged PSD of epsilon_clean.

If k_bar increases as E_clean decreases (training progresses), the residual
errors are migrating to higher wavenumbers — confirming the spectral shift
hypothesis for why B^own decreases with better models.

Two output panels:
  Left:  k_bar vs sigma, one curve per checkpoint
  Right: k_bar vs mean E_clean, one point per (checkpoint, sigma)
         — the direct test: does lower E_clean correlate with higher k_bar?

Usage:
  python experiments/eval_spectral_centroid.py \
      --checkpoint_dir checkpoints/KolmogorovFlow/forecasting/DiffusionModel_inverseCosLog-1.875_20_12 \
      --checkpoint_names epoch_501.pth epoch_1001.pth best_model.pth \
      --output results/spectral_centroid.pdf
"""
import os
import sys
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.fft import fft2, fftshift

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


# ── PSD helpers ──────────────────────────────────────────────────────────────

def radial_psd(fields: torch.Tensor) -> np.ndarray:
    """Radially-averaged PSD.  Input: (B, C, H, W).  Returns (nr,)."""
    device = fields.device
    mag = fields.pow(2).sum(dim=1).sqrt()   # (B, H, W)
    B, H, W = mag.shape
    spectrum = torch.abs(fftshift(fft2(mag), dim=(-2, -1))) ** 2
    y, x = np.indices((H, W))
    center = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r_int = torch.from_numpy(r.astype(np.int64)).to(device).view(-1)
    mag_flat = spectrum.reshape(B, -1).mean(dim=0)
    nr = min(H, W) // 2
    tbin = torch.zeros(nr + 1, device=device)
    nbin = torch.zeros(nr + 1, device=device)
    mask = r_int <= nr
    tbin.index_add_(0, r_int[mask], mag_flat[mask])
    nbin.index_add_(0, r_int[mask], torch.ones_like(mag_flat[mask]))
    return (tbin[:nr] / nbin[:nr].clamp(min=1)).cpu().numpy()


def radial_psd_1d(fields: torch.Tensor) -> np.ndarray:
    """PSD for 1-D fields.  Input: (B, C, L).  Returns (L//2,)."""
    power = torch.abs(torch.fft.fft(fields, dim=-1)) ** 2
    power = power.sum(dim=1).mean(dim=0)
    n = fields.shape[-1] // 2
    return power[:n].cpu().numpy()


def spectral_centroid(psd: np.ndarray) -> float:
    """k_bar = sum_k k * E(k) / sum_k E(k)."""
    k = np.arange(len(psd), dtype=float)
    total = psd.sum()
    if total < 1e-30:
        return float('nan')
    return float((k * psd).sum() / total)


# ── Main evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def collect(model, val_loader, device, n_batches=50, T=10):
    """
    Returns (sigmas, mean_clean, centroid_arr):
      sigmas       : (T,) noise levels, descending
      mean_clean   : (T,) mean E_clean per noise level
      centroid_arr : (T,) spectral centroid of epsilon_clean per noise level
    """
    model.eval()

    sigmas_all = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()
    T_full = len(sigmas_all)

    indices = torch.linspace(T_full - T, T_full - 1, T).long().flip(0)
    sigmas  = sigmas_all[indices]

    is_2d = None   # determined on first batch

    # Accumulators: sum of PSD and sum of E_clean over batches
    psd_acc   = [None] * T
    clean_sum = np.zeros(T)
    count     = np.zeros(T, dtype=int)

    spatial_dims = None

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        data   = sample['data'].to(device)
        cond   = data[:, 0]
        target = data[:, 1]

        if spatial_dims is None:
            spatial_dims = tuple(range(1, target.ndim))
            is_2d = (target.ndim == 4)   # (B, C, H, W)

        _, ests_clean = model(conditioning=cond, data=target,
                              return_x0_estimate=True, input_type='clean')

        for local_t, global_t in enumerate(indices.tolist()):
            est_idx   = T_full - 1 - global_t
            est_clean = ests_clean[est_idx]

            error = est_clean - target   # (B, C, ...)

            # PSD of the error field
            if is_2d:
                psd = radial_psd(error)
            else:
                psd = radial_psd_1d(error)

            if psd_acc[local_t] is None:
                psd_acc[local_t] = psd.copy()
            else:
                psd_acc[local_t] += psd

            # Scalar E_clean
            clean_mse = error.pow(2).mean(dim=spatial_dims)
            clean_sum[local_t] += clean_mse.mean().item()
            count[local_t]     += 1

    mean_clean   = clean_sum / np.maximum(count, 1)
    centroid_arr = np.array([
        spectral_centroid(psd_acc[t] / max(count[t], 1)) if psd_acc[t] is not None else float('nan')
        for t in range(T)
    ])

    return sigmas.numpy(), mean_clean, centroid_arr


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(all_results, output_path):
    """
    all_results: list of (label, sigmas, mean_clean, centroid_arr)
    """
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(all_results)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']

    fig, (ax_sigma, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: centroid vs sigma
    for (label, sigmas, _, centroid_arr), color in zip(all_results, colors):
        ax_sigma.plot(sigmas, centroid_arr, color=color, label=label, marker='o', ms=5)

    ax_sigma.set_xlabel(r'$\sigma$')
    ax_sigma.set_ylabel(r'$\bar{k}$ (spectral centroid of $\epsilon_{clean}$)')
    ax_sigma.set_title(r'Error spectral centroid vs $\sigma$')
    ax_sigma.legend()

    # Right: centroid vs E_clean — each point is a (checkpoint, sigma) pair
    # Connect same sigma across checkpoints to show training trajectory
    T = len(all_results[0][1])
    traj_colors = plt.cm.viridis(np.linspace(0.1, 0.9, T))

    for local_t in range(T):
        e_traj = [r[2][local_t] for r in all_results]   # E_clean across checkpoints
        c_traj = [r[3][local_t] for r in all_results]   # centroid across checkpoints
        sigma_val = all_results[0][1][local_t]
        ax_scatter.plot(e_traj, c_traj, '-', color=traj_colors[local_t],
                        lw=1.5, alpha=0.7)
        ax_scatter.scatter(e_traj, c_traj, s=40, color=traj_colors[local_t],
                           zorder=3, label=rf'$\sigma={sigma_val:.3f}$')

    # Mark checkpoints with different markers
    for ckpt_idx, (label, sigmas, mean_clean, centroid_arr) in enumerate(all_results):
        marker = markers[ckpt_idx % len(markers)]
        ax_scatter.scatter(mean_clean, centroid_arr,
                           marker=marker, s=60, color='k', zorder=4,
                           label=label if local_t == 0 else '')

    ax_scatter.set_xlabel(r'$E_{clean}$ (MSE, log scale)')
    ax_scatter.set_ylabel(r'$\bar{k}$ (spectral centroid)')
    ax_scatter.set_xscale('log')
    ax_scatter.set_title(r'$\bar{k}$ vs $E_{clean}$: does lower error → higher $\bar{k}$?')

    # Two separate legends: sigma colors and checkpoint markers
    from matplotlib.lines import Line2D
    sigma_handles = [Line2D([0], [0], color=traj_colors[t], lw=2,
                            label=rf'$\sigma={all_results[0][1][t]:.3f}$')
                     for t in range(T)]
    ckpt_handles  = [Line2D([0], [0], marker=markers[i % len(markers)], color='k',
                            linestyle='None', ms=7, label=all_results[i][0])
                     for i in range(len(all_results))]
    ax_scatter.legend(handles=sigma_handles + ckpt_handles,
                      fontsize=7, ncol=2, loc='best')

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved → {output_path}")

    # Summary table
    print("\n{:<20s}  {:>8s}  {:>12s}  {:>10s}".format(
        "checkpoint", "sigma", "E_clean", "k_bar"))
    for label, sigmas, mean_clean, centroid_arr in all_results:
        for t in range(len(sigmas)):
            print("{:<20s}  {:>8.4f}  {:>12.2e}  {:>10.3f}".format(
                label, sigmas[t], mean_clean[t], centroid_arr[t]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--checkpoint_names', nargs='+', default=['best_model.pth'])
    parser.add_argument('--output', default='results/spectral_centroid.pdf')
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--T', type=int, default=10,
                        help='Number of noise levels to evaluate')
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
        sigmas, mean_clean, centroid_arr = collect(
            model, val_loader, device, n_batches=args.n_batches, T=args.T)
        all_results.append((label, sigmas, mean_clean, centroid_arr))
        del model

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plot(all_results, args.output)


if __name__ == '__main__':
    main()
