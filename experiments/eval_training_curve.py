#!/usr/bin/env python
"""
Evaluate a single run at every saved epoch checkpoint and plot
clean-input error and own-prediction bias vs noise level,
with one curve per checkpoint coloured by training epoch.

Usage:
  python experiments/eval_training_curve.py \\
      --run_dir checkpoints/KuramotoSivashinsky/baselines/linear_20steps \\
      [--output results/training_curve.pdf] \\
      [--device cuda] \\
      [--n_batches 20] \\
      [--n_noise_samples 3]
"""

import os
import sys
import re
import json
import argparse
import glob as glob_module

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Style ──────────────────────────────────────────────────────────────────────
try:
    import seaborn as sns
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
except ImportError:
    pass

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.25,
})

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
    'downscale_factor', 'variables',
}

SIGMA_RANGES = [(1e-2, 5e-2), (1e-1, 1e0)]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(run_dir, ckpt_path, device):
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    from src.models.diffusion import DiffusionModel

    mp = config["model_params"]
    model = DiffusionModel(
        dimension=mp["dimension"],
        dataSize=mp["dataSize"],
        condChannels=mp["condChannels"],
        dataChannels=mp["dataChannels"],
        diffSchedule=mp["diffSchedule"],
        diffSteps=mp["diffSteps"],
        inferenceSamplingMode=mp.get("inferenceSamplingMode", "ddpm"),
        inferenceConditioningIntegration=mp.get("inferenceConditioningIntegration", "clean"),
        diffCondIntegration=mp.get("diffCondIntegration", "clean"),
        padding_mode=mp.get("padding_mode", "circular"),
        architecture=mp.get("architecture", "ours"),
        sigma_min=mp.get("sigma_min"),
        sigma_max=mp.get("sigma_max"),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, config


def get_val_loader(config):
    from src.config import DataConfig
    from src.data.loaders import get_data_loaders

    raw = config.get("data_params", {})
    data_cfg = DataConfig(**{
        **{k: v for k, v in raw.items() if k in _DATA_CONFIG_FIELDS},
        "batch_size": 256, "val_batch_size": 256,
    })
    _, val_loader, _ = get_data_loaders(data_cfg)
    return val_loader


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, val_loader, device, n_batches=20, n_noise_samples=3):
    model.eval()
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu().numpy()
    T = len(sigmas)

    all_clean = [[] for _ in range(T)]
    all_ratio = [[] for _ in range(T)]
    spatial_dims = None

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            if batch_idx >= n_batches:
                break
            data = sample["data"].to(device)
            cond, target = data[:, 0], data[:, 1]
            if spatial_dims is None:
                spatial_dims = tuple(range(1, target.ndim))

            sum_clean = sum_own = None
            for _ in range(n_noise_samples):
                _, ests_clean = model(conditioning=cond, data=target,
                                      return_x0_estimate=True, input_type="clean")
                _, ests_own = model(conditioning=cond, data=target,
                                    return_x0_estimate=True, input_type="own-pred")
                # ests are in reversed order: index 0 = t=T-1, index T-1 = t=0
                batch_clean = torch.stack(
                    [(e - target).pow(2).mean(dim=spatial_dims) for e in ests_clean])
                batch_own = torch.stack(
                    [(e - target).pow(2).mean(dim=spatial_dims) for e in ests_own])
                sum_clean = batch_clean if sum_clean is None else sum_clean + batch_clean
                sum_own   = batch_own   if sum_own   is None else sum_own   + batch_own

            avg_clean = sum_clean / n_noise_samples
            avg_own   = sum_own   / n_noise_samples

            for t_idx in range(T):
                clean_i = avg_clean[t_idx]
                own_i   = avg_own[t_idx]
                valid   = clean_i > 0
                ratio_i = (own_i / clean_i.clamp(min=1e-12))[valid]
                all_clean[t_idx].append(clean_i[valid].cpu())
                all_ratio[t_idx].append(ratio_i.cpu())

    results = {}
    for t_idx in range(T):
        if not all_clean[t_idx]:
            continue
        sigma = float(sigmas[t_idx])
        results[sigma] = {
            "clean_error":   float(torch.cat(all_clean[t_idx]).mean()),
            "own_pred_bias": float(torch.cat(all_ratio[t_idx]).mean()),
        }
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training_curves(epoch_results, output_path):
    """
    epoch_results: list of (epoch_int, {sigma: {clean_error, own_pred_bias}})
    """
    epoch_results = sorted(epoch_results, key=lambda x: x[0])
    epochs = [e for e, _ in epoch_results]

    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=min(epochs), vmax=max(epochs))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"wspace": 0.35})

    panel_labels = [r"\textbf{(a)}", r"\textbf{(b)}"]
    ylabels = [
        r"Clean-Input Error $\mathcal{E}_{\mathrm{clean}}$",
        r"Own-Pred Bias $B^{(\mathrm{own})}$",
    ]

    # Build interpolated curves on a common sigma grid for fill_between
    all_sigmas = sorted({s for _, r in epoch_results for s in r})
    common_sigmas = np.array(all_sigmas)

    def interp_vals(results, key):
        sigs = np.array(sorted(results.keys()))
        vals = np.array([results[s][key] for s in sigs])
        mask = vals > 0
        if not mask.any():
            return np.full(len(common_sigmas), np.nan)
        return np.exp(np.interp(np.log(common_sigmas),
                                np.log(sigs[mask]), np.log(vals[mask])))

    curves = [
        (epoch,
         interp_vals(results, "clean_error"),
         interp_vals(results, "own_pred_bias"))
        for epoch, results in epoch_results
    ]

    for i, (epoch, clean, bias) in enumerate(curves):
        color = cmap(norm(epoch))
        for ax, vals in zip(axes, [clean, bias]):
            valid = ~np.isnan(vals) & (common_sigmas < 1.2e-1)
            if not valid.any():
                continue
            ax.plot(common_sigmas[valid], vals[valid][::-1],
                    color=color, linewidth=2.5, alpha=0.9, zorder=3)


    sigma_min = common_sigmas[0]
    sigma_max = min(common_sigmas[common_sigmas < 1.2e-1].max(), 1.2e-1)

    for ax, ylabel, panel in zip(axes, ylabels, panel_labels):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Noise Level $\sigma$", labelpad=6)
        ax.set_xlim(sigma_min, sigma_max)
        ax.set_xticks([sigma_min, sigma_max])
        ax.set_xticklabels([rf"${sigma_min:.1e}$", rf"${sigma_max:.1e}$"])
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_title(panel, loc="left", pad=6)
        # Spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("0.3")
        ax.tick_params(which="both", direction="in", length=4, color="0.3")
        ax.tick_params(which="minor", length=2)
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, color="0.88")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="0.93")
        ax.set_axisbelow(True)


    fig.subplots_adjust(right=0.87)
    cax = fig.add_axes([0.90, 0.13, 0.018, 0.74])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Training Epoch", labelpad=10)
    cbar.ax.tick_params(labelsize=10, direction="in")
    cbar.outline.set_linewidth(0.8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def find_epoch_checkpoints(run_dir):
    """Return sorted list of (epoch, path) for all epoch_*.pth files."""
    pattern = os.path.join(run_dir, "epoch_*.pth")
    paths = glob_module.glob(pattern)
    result = []
    for p in paths:
        m = re.search(r"epoch_(\d+)\.pth$", p)
        if m:
            result.append((int(m.group(1)), p))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                        help="Directory containing config.json and epoch_*.pth checkpoints.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--n_noise_samples", type=int, default=3)
    parser.add_argument("--output", default=None,
                        help="Output PDF path. Defaults to <run_dir>/training_curve.pdf.")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.run_dir, "training_curve.pdf")
    cache_path  = output_path.replace(".pdf", "_cache.json")

    epoch_ckpts = find_epoch_checkpoints(args.run_dir)
    if not epoch_ckpts:
        raise FileNotFoundError(f"No epoch_*.pth files found in {args.run_dir}")
    n_ckpts = min(10, len(epoch_ckpts))
    indices = [round(i * (len(epoch_ckpts) - 1) / (n_ckpts - 1)) for i in range(n_ckpts)]
    epoch_ckpts = [epoch_ckpts[i] for i in indices]
    print(f"Evaluating {n_ckpts} checkpoints: epochs {[e for e,_ in epoch_ckpts]}")

    if os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        with open(cache_path) as f:
            raw = json.load(f)
        epoch_results = [(entry["epoch"], {float(k): v for k, v in entry["results"].items()})
                         for entry in raw]
    else:
        # Load val_loader once (all checkpoints share the same config)
        _, config = load_model(args.run_dir, epoch_ckpts[0][1], args.device)
        val_loader = get_val_loader(config)

        epoch_results = []
        for i, (epoch, ckpt_path) in enumerate(epoch_ckpts):
            print(f"[{i+1}/{len(epoch_ckpts)}] Evaluating epoch {epoch} ...")
            model, _ = load_model(args.run_dir, ckpt_path, args.device)
            model.compute_schedule_variables(model.sqrtOneMinusAlphasCumprod.ravel()[-20:])
            results = evaluate_model(model, val_loader, args.device,
                                     n_batches=args.n_batches,
                                     n_noise_samples=args.n_noise_samples)
            epoch_results.append((epoch, results))
            print(f"           {len(results)} sigma levels evaluated.")
            del model

        with open(cache_path, "w") as f:
            json.dump([{"epoch": e, "results": {str(k): v for k, v in r.items()}}
                       for e, r in epoch_results], f, indent=2)
        print(f"Cached results to {cache_path}")

    plot_training_curves(epoch_results, output_path)


if __name__ == "__main__":
    main()
