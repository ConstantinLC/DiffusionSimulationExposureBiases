#!/usr/bin/env python
"""
Plot instability threshold curves gamma(sigma, tau) for multiple datasets.

Reads error_tracking_map.json files produced by train_diffusion_model with
track_instability=True and generates Figure 2 of the paper: for each
(sigma, tau) pair, the clean-input error E_clean at which the own-prediction
ratio B^(own) first dropped below tau.

This validates Hypothesis 3.1:
  - gamma(sigma, tau) is increasing in sigma  (curves go up to the right)
  - gamma(sigma, tau) is decreasing in tau    (stricter tau => lower threshold)

Usage:
  python experiments/plot_instability_thresholds.py \
      --maps kolmo=checkpoints/KolmogorovFlow/instability_threshold/*/error_tracking_map.json \
             ks=checkpoints/KuramotoSivashinsky/instability_threshold/*/error_tracking_map.json \
             tra=checkpoints/TransonicFlow/instability_threshold/*/error_tracking_map.json \
      --output results/instability_thresholds_all_datasets.pdf
"""
import os
import sys
import json
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
}


def compute_spectral_flatness(map_path: str, map_data: dict, device: str = 'cpu') -> float:
    """
    Compute the spectral flatness of the model's prediction error:

        phi = mean_k P(k) / max_k P(k)

    where P(k) = mean_{samples, channels} |FFT(y_hat - y)[k]|^2.

    The spectral *shape* of the prediction error is constant across diffusion
    timesteps — only its amplitude scales with sigma. Therefore phi is a single
    scalar for the whole model/dataset, computed at one representative timestep
    (the median sigma level in map_data).

    phi = 1   -> white error (all modes equally powerful)
    phi = 1/d -> error concentrated in a single mode

    Returns a float, or None on failure.
    """
    config_path = os.path.join(os.path.dirname(map_path), 'config.json')
    ckpt_path   = os.path.join(os.path.dirname(map_path), 'best_model.pth')
    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        print(f"  [flatness] config.json or best_model.pth not found, skipping.")
        return None
    try:
        import torch
        from src.config import DataConfig
        from src.data.loaders import get_data_loaders
        from src.models.diffusion import DiffusionModel

        with open(config_path) as f:
            config = json.load(f)

        mp = config['model_params']
        model = DiffusionModel(
            dimension=mp['dimension'],
            dataSize=mp['dataSize'],
            condChannels=mp['condChannels'],
            dataChannels=mp['dataChannels'],
            diffSchedule=mp['diffSchedule'],
            diffSteps=mp['diffSteps'],
            inferenceSamplingMode=mp['inferenceSamplingMode'],
            inferenceConditioningIntegration=mp['inferenceConditioningIntegration'],
            diffCondIntegration=mp['diffCondIntegration'],
            padding_mode=mp.get('padding_mode', 'circular'),
            architecture=mp.get('architecture', 'ours'),
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        raw = config.get('data_params', {})
        data_cfg = DataConfig(**{
            **{k: v for k, v in raw.items() if k in _DATA_CONFIG_FIELDS},
            'batch_size': 256, 'val_batch_size': 256,
        })
        _, val_loader, _ = get_data_loaders(data_cfg)

        # Pick the median sigma level as the representative timestep
        sigma_values = sorted([float(k) for k in map_data.keys()])
        sigma_ref = float(np.median(sigma_values))
        all_sigmas = model.sqrtOneMinusAlphasCumprod.ravel().cpu().numpy()
        t_idx = int(np.argmin(np.abs(all_sigmas - sigma_ref)))
        alpha_t = model.sqrtAlphasCumprod.ravel()[t_idx]
        sigma_t = model.sqrtOneMinusAlphasCumprod.ravel()[t_idx]
        t_scalar = torch.tensor(t_idx, device=device)
        print(f"  [flatness] Computing at sigma={all_sigmas[t_idx]:.4f} (t={t_idx})")

        psd_acc = None
        n_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(device)
                cond, y = data[:, 0], data[:, 1]
                B = y.shape[0]

                eps = torch.randn_like(y)
                t_batch = t_scalar.expand(B)
                y_noisy = alpha_t * y + sigma_t * eps

                inp = torch.cat([cond, y_noisy], dim=1)
                pred_noise = model.unet(inp, t_batch)[:, cond.shape[1]:]
                y_hat = (y_noisy - sigma_t * pred_noise) / alpha_t

                delta = (y_hat - y).float()
                if delta.ndim == 3:
                    fft = torch.fft.rfft(delta, dim=-1, norm='ortho')
                else:
                    fft = torch.fft.rfft2(delta, dim=(-2, -1), norm='ortho')

                power = fft.abs().pow(2).mean(dim=(0, 1)).reshape(-1)
                psd_acc = power * B if psd_acc is None else psd_acc + power * B
                n_samples += B

        psd = (psd_acc / n_samples).cpu().numpy()
        phi = float(psd.mean() / psd.max()) if psd.max() > 0 else 1.0
        print(f"  [flatness] phi={phi:.4e}  (max={psd.max():.2e}  mean={psd.mean():.2e}  "
              f"d={len(psd)})")
        return phi

    except Exception as e:
        print(f"  [flatness] Failed: {e}")
        import traceback; traceback.print_exc()
        return None

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
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.25,
}
mpl.rcParams.update(rcParams)

def _log_power_scale_funcs(p):
    """
    Custom axis scale  f(σ) = −(−log₁₀ σ)^p  for σ ∈ (0, 1].

    p = 1  →  standard log₁₀ scale (equal spacing in log σ).
    p > 1  →  progressively more axis space given to small σ.

    The forward map is strictly increasing in σ (small σ → large negative value,
    σ = 1 → 0), so left-to-right ordering is preserved.
    """
    def forward(x):
        x = np.asarray(x, dtype=float)
        neg_log = -np.log10(np.clip(x, 1e-15, 1.0))  # ≥ 0
        return -(neg_log ** p)

    def inverse(y):
        y = np.asarray(y, dtype=float)
        neg_log = (-np.clip(y, None, 0.0)) ** (1.0 / p)
        return 10.0 ** (-neg_log)

    return forward, inverse


DATASET_LABELS = {
    "kolmo": "Kolmogorov Flow",
    "ks": "Kuramoto-Sivashinsky",
    "tra": "Transonic Flow",
}


def load_map(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def gamma_theory(sigma, tau, phi=1.0):
    """
    Theoretical instability threshold (Remark H.6):
        gamma(sigma, tau) = (tau - 1) * phi * sigma^2 / (1 - sigma^2)

    phi is the spectral flatness of the prediction error:
        phi = mean_k P(k) / max_k P(k)  in [1/d, 1]

    phi = 1   -> white error (Parseval / L2 bound, overestimates gamma)
    phi = 1/d -> error concentrated in one mode (tightest bound)
    """
    sigma = np.asarray(sigma)
    return (tau - 1.0) * phi * sigma ** 2 / (1.0 - sigma ** 2)


def plot_single_dataset(ax, data: dict, title: str, show_ylabel=True, show_legend=True,
                        x_scale_power=1.0, phi_dict=None):
    """Plot gamma(sigma, tau) curves for a single dataset."""
    x_values = sorted([float(k) for k in data.keys()])
    all_taus = sorted(set(t for sub in data.values() for t in sub.keys()))

    colors = plt.cm.plasma(np.linspace(0, 0.85, len(all_taus)))
    SIGMA_SMOOTHING = 2.0

    # Collect all y-values to set tight axis limits after plotting
    all_ys = []
    all_xs_plotted = []

    for idx, tau_str in enumerate(all_taus):
        xs, ys = [], []
        for x in x_values:
            key = next((k for k in data.keys() if abs(float(k) - x) < 1e-9), None)
            if key and tau_str in data[key]:
                xs.append(x)
                ys.append(data[key][tau_str]["clean_error"])

        if len(xs) > 1:
            label = rf"$\tau = {tau_str}$"
            ax.plot(xs, ys, label=label, color=colors[idx], alpha=1.0, zorder=idx + 10)
            y_floor = min(ys) * 0.5
            ax.fill_between(xs, ys, y_floor, color=colors[idx], alpha=0.12, zorder=idx)
            all_ys.extend(ys)
            all_xs_plotted.extend(xs)

            # Theoretical bound with spectral flatness correction (Remark H.6)
            tau_val = float(tau_str)
            xs_arr = np.array(xs)
            phi = phi_dict if phi_dict is not None else 1.0
            gamma = gamma_theory(xs_arr, tau_val, phi=phi)
            ax.plot(xs_arr, gamma, color=colors[idx], alpha=0.5, linestyle='--',
                    linewidth=1.2, zorder=idx + 5)



    # Add a single proxy artist for the theoretical bound in the legend (Corollary H.5)
    if all_xs_plotted:
        ax.plot([], [], color='gray', linestyle='--', linewidth=1.2,
                label=r'$(\tau{-}1)\,C(P,\sigma)\,\mathrm{SNR}_\mathrm{diff}^{-1}$')

    # Tight axis limits based on actual data
    if all_ys and all_xs_plotted:
        y_min = min(all_ys)
        y_max = max(all_ys)
        x_min = min(all_xs_plotted)
        x_max = max(all_xs_plotted)
        margin_y = 0.5  # one half-decade of padding
        ax.set_ylim(y_min * 10**(-margin_y), y_max * 10**margin_y)
        ax.set_xlim(x_min, x_max)

    # X-axis scale
    if x_scale_power == 1.0:
        ax.set_xscale('log')
    else:
        fw, inv = _log_power_scale_funcs(x_scale_power)
        ax.set_xscale('function', functions=(fw, inv))
        # Place ticks at round sigma values within data range
        candidate_ticks = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05,
                                     0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])
        x_min_plot = min(all_xs_plotted) if all_xs_plotted else 0.01
        x_max_plot = max(all_xs_plotted) if all_xs_plotted else 1.0
        ticks_raw = candidate_ticks[(candidate_ticks >= x_min_plot) & (candidate_ticks <= x_max_plot)]
        # Remove ticks that are too close in the transformed axis space
        if len(ticks_raw) > 1:
            t_vals = fw(ticks_raw)
            total_span = t_vals[-1] - t_vals[0]
            min_gap = 0.05 * total_span  # 5% of axis width minimum spacing
            keep = [True] * len(ticks_raw)
            for i in range(1, len(ticks_raw)):
                if keep[i - 1] and (t_vals[i] - t_vals[i - 1]) < min_gap:
                    keep[i] = False
            ticks = ticks_raw[np.array(keep)]
        else:
            ticks = ticks_raw
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: (f'${v:.3g}$' if v < 0.1 else f'${v:.2g}$')
        ))
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.set_yscale('log')
    ax.set_xlabel(r"Noise Level $\sigma$", labelpad=8)
    if show_ylabel:
        ax.set_ylabel(r"Clean-Input Error $\mathcal{E}_{\mathrm{clean}}$", labelpad=8)
    ax.set_title(rf"\textbf{{{title}}}", pad=10)
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='0.85')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='0.9')
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(title=r"\textbf{Instability Thresholds}", loc='upper left',
                  frameon=True, fancybox=False, framealpha=0.95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', nargs='+', required=True,
                        help='key=path pairs, e.g. kolmo=path/to/map.json')
    parser.add_argument('--output', default='results/instability_thresholds_all.pdf')
    parser.add_argument('--x_scale_power', type=float, default=2.0,
                        help='Power p for -(−log10 σ)^p x-axis scale. '
                             '1=standard log, 2=more emphasis on small σ.')
    parser.add_argument('--no_flatness', action='store_true',
                        help='Skip spectral flatness computation and use phi=1 '
                             '(the white-error / L2 bound).')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Parse key=path arguments, storing both the map data and the resolved file path
    datasets = {}       # key -> map dict
    map_paths = {}      # key -> resolved file path
    for entry in args.maps:
        key, path = entry.split('=', 1)
        matches = sorted(glob(path))
        if matches:
            datasets[key] = load_map(matches[0])
            map_paths[key] = matches[0]
        else:
            print(f"Warning: no file matched for {key}: {path}")

    if not datasets:
        print("No data loaded. Exiting.")
        return

    # Compute spectral flatness phi per dataset (single scalar — shape is sigma-invariant)
    phi_dicts = {}
    for key, map_path in map_paths.items():
        if args.no_flatness:
            phi_dicts[key] = None
        else:
            phi_dicts[key] = compute_spectral_flatness(map_path, datasets[key])

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)

    for idx, (key, data) in enumerate(datasets.items()):
        title = DATASET_LABELS.get(key, key)
        plot_single_dataset(
            axes[0, idx], data, title,
            show_ylabel=(idx == 0),
            show_legend=(idx == 0),
            x_scale_power=args.x_scale_power,
            phi_dict=phi_dicts[key],
        )

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == '__main__':
    main()
