#!/usr/bin/env python
"""
Plot instability threshold curves gamma(sigma, tau) for a single run.

Reads error_tracking_map.json produced by train_diffusion_model with
track_instability=True and generates Figure 2 of the paper: for each
(sigma, tau) pair, the clean-input error E_clean at which the own-prediction
ratio B^(own) first dropped below tau.

This validates Hypothesis 3.1:
  - gamma(sigma, tau) is increasing in sigma  (curves go up to the right)
  - gamma(sigma, tau) is decreasing in tau    (stricter tau => lower threshold)

Usage:
  python experiments/plot_instability_thresholds.py \
      --run checkpoints/KolmogorovFlow/instability_threshold/run_001
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
    "kolmogorovflow": "Kolmogorov Flow",
    "kuramotosivashinsky": "Kuramoto-Sivashinsky",
    "transonicflow": "Transonic Flow",
    "weatherbench": "WeatherBench",
}


def load_map(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_one(ax, data: dict, label: str = None, show_ylabel=True, show_legend=True):
    """Plot gamma(sigma, tau) curves for a single dataset onto ax."""
    x_values = sorted([float(k) for k in data.keys()])
    all_taus = sorted(set(t for sub in data.values() for t in sub.keys()))
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(all_taus)))

    all_ys = []
    all_xs_plotted = []

    curves = []
    for tau_str in all_taus:
        xs, ys = [], []
        for x in x_values:
            key = next((k for k in data.keys() if abs(float(k) - x) < 1e-9), None)
            if key and tau_str in data[key]:
                entry = data[key][tau_str]
                if "clean_errors" in entry:
                    val = float(np.mean(entry["clean_errors"]))
                else:
                    val = entry["clean_error"]
                xs.append(x)
                ys.append(val)
        if len(xs) > 1:
            curves.append((tau_str, xs, ys))
            all_ys.extend(ys)
            all_xs_plotted.extend(xs)

    # For each sigma, collect which taus have data (to decide red-cross condition)
    sigma_to_taus = {}
    for tau_str, xs, ys in curves:
        for x in xs:
            sigma_to_taus.setdefault(x, set()).add(float(tau_str))

    for tau_str, xs, ys in curves:
        tau_idx = all_taus.index(tau_str)
        ys_arr = np.array(ys)
        if len(ys_arr) >= 3:
            ys_smooth = np.exp(gaussian_filter1d(np.log(ys_arr), sigma=1.5))
        else:
            ys_smooth = ys_arr
        ax.plot(xs, ys_smooth, label=rf"$\gamma(\sigma, {tau_str})$",
                color=colors[tau_idx], alpha=1.0, zorder=tau_idx + 10)
        y_floor = min(ys) * 0.5
        ax.fill_between(xs, ys_smooth, y_floor, color=colors[tau_idx], alpha=0.12, zorder=tau_idx)
        # Red cross at leftmost point if no smaller tau has data at that sigma
        leftmost_sigma = xs[0]
        smaller_taus_at_sigma = [t for t in sigma_to_taus.get(leftmost_sigma, set())
                                  if t < float(tau_str)]
        if not smaller_taus_at_sigma:
            ax.plot(leftmost_sigma, ys_smooth[0], marker='x', color='red', markersize=8,
                    markeredgewidth=2, zorder=tau_idx + 20, linestyle='none')

    if all_ys and all_xs_plotted:
        y_min, y_max = min(all_ys), max(all_ys)
        x_min, x_max = min(all_xs_plotted), max(all_xs_plotted)
        ax.set_ylim(y_min * 10**(-0.5), y_max * 10**0.5)
        ax.set_xlim(x_min, x_max)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"Noise Level $\sigma$", labelpad=8)
    if show_ylabel:
        ax.set_ylabel(r"Clean-Input Error $\mathcal{E}_{\mathrm{clean}}$", labelpad=8)
    if label:
        ax.set_title(rf"\textbf{{{label}}}", pad=8)
    if all_xs_plotted:
        x_min_tick = min(all_xs_plotted)
        x_max_tick = max(all_xs_plotted)
        ax.set_xticks([x_min_tick, x_max_tick])
        def _log_fmt(v, _):
            exp = int(np.floor(np.log10(v)))
            mantissa = v / 10**exp
            if abs(mantissa - 1.0) < 0.01:
                return rf'$10^{{{exp}}}$'
            return rf'${mantissa:.2g}{{\times}}10^{{{exp}}}$'
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_fmt))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='0.85')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='0.9')
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(title=r"\textbf{Stability Thresholds}", loc='upper left',
                  frameon=True, fancybox=False, framealpha=0.95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, nargs='+',
                        help='One or two run directories containing error_tracking_map.json')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Optional subplot titles for each run')
    args = parser.parse_args()

    if len(args.run) > 2:
        print("Error: at most 2 runs are supported")
        return

    datasets = []
    for run in args.run:
        run_path = os.path.abspath(run)
        map_path = os.path.join(run_path, 'error_tracking_map.json')
        if not os.path.exists(map_path):
            print(f"Error: no error_tracking_map.json found in {run_path}")
            return
        datasets.append(load_map(map_path))

    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4.5 * n), squeeze=False)

    for i, (ax, data) in enumerate(zip(axes[:, 0], datasets)):
        lbl = args.labels[i] if args.labels and i < len(args.labels) else None
        plot_one(ax, data, label=lbl, show_ylabel=True, show_legend=True)

    plt.tight_layout()
    output = os.path.join(os.path.abspath(args.run[0]), 'instability_thresholds.pdf')
    plt.savefig(output, bbox_inches='tight')
    print(f"Saved to {output}")
    plt.close(fig)


if __name__ == '__main__':
    main()
