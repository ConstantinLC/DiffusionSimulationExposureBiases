#!/usr/bin/env python
"""
Experiment H.5 — Spectral flooding investigation.

For a trained diffusion model at diverse noise levels sigma_k, compute:
  1. The PSD of the clean-input reconstruction error  epsilon_clean^k = y_hat^k - y
  2. The PSD of the diffusion noise floor  sqrt(1 - alpha_k) * z
  3. The own-prediction perturbation  sqrt(alpha_k) * epsilon_clean^k

Then check the spectral masking condition (Eq. 43 from the paper):
  Does the noise floor flood the error at every wavenumber?
  i.e.  alpha_sigma * |delta_hat(k)|^2 <= sigma^2  for all k

Usage:
  python experiments/eval_spectral_flooding.py \
      --checkpoint_dir checkpoints/KolmogorovFlow/forecasting/DiffusionModel_linear_20 \
      --output results/spectral_flooding.pdf

  # Specify noise-level indices to plot (default: ~6 spread across schedule):
  python experiments/eval_spectral_flooding.py \
      --checkpoint_dir checkpoints/KolmogorovFlow/forecasting/DiffusionModel_linear_20 \
      --noise_indices 0 4 9 14 19 \
      --output results/spectral_flooding.pdf
"""
import os
import sys
import json
import argparse

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

# ── Matplotlib ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def radial_psd(fields: torch.Tensor) -> np.ndarray:
    """
    Compute the radially-averaged power spectral density of a batch of 2-D
    fields.  Input shape: (B, C, H, W).  Returns array of shape (nr,).
    """
    device = fields.device
    # Channel-wise L2 norm -> (B, H, W)
    mag = fields.pow(2).sum(dim=1).sqrt()
    B, H, W = mag.shape

    spectrum = torch.abs(fftshift(fft2(mag), dim=(-2, -1))) ** 2

    # Radial binning
    y, x = np.indices((H, W))
    center = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r_int = torch.from_numpy(r.astype(np.int64)).to(device).view(-1)

    mag_flat = spectrum.reshape(B, -1).mean(dim=0)  # average over batch
    nr = min(H, W) // 2

    tbin = torch.zeros(nr + 1, device=device)
    nbin = torch.zeros(nr + 1, device=device)

    mask = r_int <= nr
    tbin.index_add_(0, r_int[mask], mag_flat[mask])
    nbin.index_add_(0, r_int[mask], torch.ones_like(mag_flat[mask]))

    profile = (tbin[:nr] / nbin[:nr].clamp(min=1)).cpu().numpy()
    return profile


def radial_psd_1d(fields: torch.Tensor) -> np.ndarray:
    """PSD for 1-D fields.  Input shape: (B, C, L).  Returns array of shape (L//2,)."""
    power = torch.abs(torch.fft.fft(fields, dim=-1)) ** 2
    power = power.sum(dim=1).mean(dim=0)  # sum channels, mean batch
    n = fields.shape[-1] // 2
    return power[:n].cpu().numpy()


# ── Main evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, n_batches=20):
    """
    For every noise level in the model's schedule, accumulate:
      - PSD of the clean-input reconstruction error (epsilon_clean)
      - PSD of Gaussian noise (reference noise floor)
      - PSD of the target signal
    Returns dict mapping t_idx -> results.
    """
    model.eval()

    T = 20
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()[-T:]
    alphas_cumprod = model.sqrtAlphasCumprod.squeeze().cpu()[-T:] ** 2  # alpha_bar
    

    # Accumulators: one per noise level
    error_psd_acc = [0.0 for _ in range(T)]
    own_error_psd_acc = [0.0 for _ in range(T)]
    noise_psd_acc = [0.0 for _ in range(T)]
    mse_clean_acc = [0.0 for _ in range(T)]
    mse_own_acc = [0.0 for _ in range(T)]
    signal_psd_acc = 0.0
    count = 0

    is_2d = model.dimension == 2
    psd_fn = radial_psd if is_2d else radial_psd_1d

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        data = sample['data'].to(device)
        cond = data[:, 0]
        target = data[:, 1]

        # Signal PSD (accumulated once, not per noise level)
        signal_psd_acc = signal_psd_acc + psd_fn(target)

        # Clean-input forward: get x0 estimates at every noise level
        _, x0_clean = model(
            conditioning=cond, data=target,
            return_x0_estimate=True, input_type='clean',
        )
        # Own-prediction forward
        _, x0_own = model(
            conditioning=cond, data=target,
            return_x0_estimate=True, input_type='own-pred',
        )
        # x0_estimates is a list of length T (high sigma -> low sigma)

        spatial_dims = tuple(range(1, target.ndim))
        for t_idx in range(T):
            # Reconstruction errors
            error_clean = x0_clean[t_idx] - target
            error_own = x0_own[t_idx] - target
            error_psd_acc[t_idx] = error_psd_acc[t_idx] + psd_fn(error_clean)
            own_error_psd_acc[t_idx] = own_error_psd_acc[t_idx] + psd_fn(error_own)
            mse_clean_acc[t_idx] += error_clean.pow(2).mean(dim=spatial_dims).mean().item()
            mse_own_acc[t_idx] += error_own.pow(2).mean(dim=spatial_dims).mean().item()

            # Reference noise sample at this noise level
            sigma_t = sigmas[T - 1 - t_idx]  # x0_estimates[0] corresponds to highest sigma
            noise_sample = sigma_t * torch.randn_like(target)
            noise_psd_acc[t_idx] = noise_psd_acc[t_idx] + psd_fn(noise_sample)

        count += 1

    # Average
    signal_psd = signal_psd_acc / count
    error_psds = [acc / count for acc in error_psd_acc]
    own_error_psds = [acc / count for acc in own_error_psd_acc]
    noise_psds = [acc / count for acc in noise_psd_acc]
    mse_clean = np.array([acc / count for acc in mse_clean_acc])
    mse_own = np.array([acc / count for acc in mse_own_acc])

    return {
        'signal_psd': signal_psd,
        'error_psds': error_psds,       # indexed [t_idx] where 0 = highest sigma
        'own_error_psds': own_error_psds,
        'noise_psds': noise_psds,
        'mse_clean': mse_clean,         # indexed [t_idx] where 0 = highest sigma
        'mse_own': mse_own,
        'sigmas': sigmas.numpy(),
        'alphas_cumprod': alphas_cumprod.numpy(),
        'T': T,
    }


def plot_results(all_results, noise_indices, output_path):
    """
    Overlay multiple snapshot epochs.  all_results is an OrderedDict
    mapping snapshot_name -> results dict.

    For each selected noise level:
      - Top row: PSD curves (perturbation vs noise floor) for each snapshot
      - Bottom row: per-wavenumber own-pred / clean ratio for each snapshot
    """
    # Use first snapshot for shared quantities (sigmas, signal_psd, noise_psd)
    ref = next(iter(all_results.values()))
    T = ref['T']
    sigmas = ref['sigmas']
    alphas_cumprod = ref['alphas_cumprod']

    snapshot_names = list(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshot_names)))

    n_plots = len(noise_indices)
    fig, axes = plt.subplots(2, n_plots, figsize=(5 * n_plots, 9), squeeze=False)

    freqs = np.arange(len(ref['signal_psd']))

    for ax_idx, t_idx in enumerate(noise_indices):
        ax_top = axes[0, ax_idx]
        ax_bot = axes[1, ax_idx]

        est_idx = T - 1 - t_idx

        sigma = sigmas[t_idx]
        alpha_sigma = 1.0 - sigma ** 2

        # Noise floor (shared across snapshots — same schedule, same sigma)
        noise_psd = ref['noise_psds'][est_idx]

        # ── Top row: PSD curves ──
        ax_top.semilogy(freqs, ref['signal_psd'], 'k-', alpha=0.2, label='Signal PSD')
        ax_top.semilogy(freqs, noise_psd, 'r--', alpha=0.6, label=r'$\sigma^2$ noise floor')

        for snap_idx, (snap_name, results) in enumerate(all_results.items()):
            c = colors[snap_idx]
            error_psd = results['error_psds'][est_idx]
            perturbation_psd = alpha_sigma * error_psd
            ax_top.semilogy(freqs, perturbation_psd, color=c, linewidth=2.0,
                            label=rf'{snap_name}: $\alpha_\sigma |\hat\delta|^2$')

        ax_top.set_title(rf'$\sigma = {sigma:.3f}$')
        if ax_idx == 0:
            ax_top.set_ylabel('Power')
        ax_top.legend(fontsize=7, loc='upper right')
        ax_top.grid(True, which='both', alpha=0.3, linestyle=':')

        # ── Bottom row: own-pred / clean error ratio per wavenumber ──
        ax_bot.axhline(1.0, color='red', linestyle='--', alpha=0.7, label=r'$r_k = 1$')

        title_parts = []
        for snap_idx, (snap_name, results) in enumerate(all_results.items()):
            c = colors[snap_idx]
            own_error_psd = results['own_error_psds'][est_idx]
            error_psd = results['error_psds'][est_idx]
            ratio_psd = own_error_psd / (error_psd + 1e-30)
            ax_bot.plot(freqs, ratio_psd, color=c, linewidth=1.8, label=snap_name)

            # Total scalar ratio
            e_clean = results['mse_clean'][est_idx]
            e_own = results['mse_own'][est_idx]
            total_ratio = e_own / (e_clean + 1e-30)
            title_parts.append(rf'{snap_name}: {total_ratio:.3f}')

        ax_bot.set_xlabel('Wavenumber $k$')
        if ax_idx == 0:
            ax_bot.set_ylabel(r'$r_k$ (per-mode bias ratio)')
        ax_bot.set_title(r'$\mathcal{B}^{own}$: ' + ', '.join(title_parts), fontsize=9)
        ax_bot.legend(fontsize=7, loc='upper left')
        ax_bot.grid(True, which='both', alpha=0.3, linestyle=':')

    fig.suptitle('H.5: Spectral masking — does the noise floor flood the error?',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # ── Summary figure: ratio at each noise level (one curve per snapshot) ──
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for snap_idx, (snap_name, results) in enumerate(all_results.items()):
        c = colors[snap_idx]
        all_sigmas = []
        frac_violated = []
        max_ratio = []

        for t_idx in range(T):
            est_idx = T - 1 - t_idx
            sigma = sigmas[t_idx]
            alpha_sigma = 1.0 - sigma ** 2

            error_psd = results['error_psds'][est_idx]
            noise_psd = ref['noise_psds'][est_idx]
            perturbation_psd = alpha_sigma * error_psd

            ratio = perturbation_psd / (noise_psd + 1e-30)

            all_sigmas.append(sigma)
            frac_violated.append(np.mean(ratio > 1.0))
            max_ratio.append(np.max(ratio))

        all_sigmas = np.array(all_sigmas)
        ax1.plot(all_sigmas, frac_violated, 'o-', color=c, label=snap_name)
        ax2.semilogy(all_sigmas, max_ratio, 'o-', color=c, label=snap_name)

    ax1.set_xlabel(r'Noise level $\sigma$')
    ax1.set_ylabel('Fraction of wavenumbers\nwhere masking violated')
    ax1.set_xscale('log')
    ax1.set_title('Masking condition violations')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.axhline(0, color='green', linestyle='--', alpha=0.5)
    ax1.legend()

    ax2.set_xlabel(r'Noise level $\sigma$')
    ax2.set_ylabel(r'$\max_k \; \alpha_\sigma |\hat\delta(k)|^2 / \sigma^2$')
    ax2.set_xscale('log')
    ax2.set_title(r'Worst-case per-mode ratio')
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Masking boundary')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    summary_path = output_path.replace('.pdf', '_summary.pdf').replace('.png', '_summary.png')
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    print(f"Summary saved to {summary_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

SNAPSHOT_FILES = [
    ('epoch 500',  'epoch_501.pth'),
    ('epoch 1000', 'epoch_1001.pth'),
    ('best',       'best_model.pth'),
]


def main():
    parser = argparse.ArgumentParser(description='H.5: Spectral flooding investigation')
    parser.add_argument('--checkpoint_dir', required=True,
                        help='Path to checkpoint directory (must contain config.json)')
    parser.add_argument('--noise_indices', type=int, nargs='+', default=None,
                        help='Schedule indices to plot (0 = lowest sigma). '
                             'Default: ~6 evenly spaced.')
    parser.add_argument('--n_batches', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/spectral_flooding.pdf')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    # Build data loader (shared across snapshots)
    raw_data = config['data_params']
    data_cfg = DataConfig(**{
        k: v for k, v in raw_data.items() if k in _DATA_CONFIG_FIELDS
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    # Evaluate each snapshot
    from collections import OrderedDict
    all_results = OrderedDict()

    for snap_name, ckpt_file in SNAPSHOT_FILES:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {snap_name}: {ckpt_path} not found")
            continue

        model = build_diffusion_model(config['model_params']).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        print(f"Loaded {snap_name} from {ckpt_path}")

        results = evaluate(model, val_loader, args.device, n_batches=args.n_batches)
        all_results[snap_name] = results

        del model
        torch.cuda.empty_cache()

    if not all_results:
        print("No checkpoints found, exiting.")
        return

    T = next(iter(all_results.values()))['T']
    sigmas = next(iter(all_results.values()))['sigmas']
    print(f"Schedule: T={T}, sigma range [{sigmas.min():.4f}, {sigmas.max():.4f}]")

    # Select noise indices to plot
    if args.noise_indices is not None:
        noise_indices = args.noise_indices
    else:
        noise_indices = np.linspace(0, T - 1, min(6, T), dtype=int).tolist()

    # Plot
    plot_results(all_results, noise_indices, args.output)


if __name__ == '__main__':
    main()
