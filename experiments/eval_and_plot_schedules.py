#!/usr/bin/env python
"""
Evaluate 6 converged models (trained with different log-uniform sigma ranges)
and plot clean error + own-prediction bias vs noise level.

Three groups:
  Group 1 (1 model):  model1_full            — full range [sigma_min, sigma_max]
  Group 2 (2 models): model2_lower_half,
                      model3_upper_half       — each covers half the range
  Group 3 (3 models): model4_lower_third,
                      model5_mid_third,
                      model6_upper_third      — each covers a third of the range

For each group we stitch the per-model results into one curve spanning the
full sigma range and plot:
  Left panel  — clean-input MSE  E_clean(sigma)
  Right panel — own-prediction bias  B^(own)(sigma) = E_own / E_clean

Usage:
  python experiments/eval_and_plot_schedules.py \\
      --base_dir checkpoints/KuramotoSivashinsky/instability_schedules

  # or with explicit per-model run dirs (innermost dirs containing best_model.pth)
  python experiments/eval_and_plot_schedules.py \\
      --runs path/to/model1 path/to/model2 path/to/model3 \\
             path/to/model4 path/to/model5 path/to/model6
"""

import os
import sys
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

# ── Model loading ──────────────────────────────────────────────────────────────

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
}


def find_run_dir(parent: str) -> str:
    """Find the innermost run dir containing best_model.pth inside *parent*."""
    if os.path.exists(os.path.join(parent, "best_model.pth")):
        return parent
    candidates = sorted(glob_module.glob(os.path.join(parent, "*", "best_model.pth")))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pth found under {parent}")
    return os.path.dirname(candidates[-1])  # take the most recent if multiple


def load_model_and_data(run_dir: str, device: str):
    config_path = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, "best_model.pth")

    with open(config_path) as f:
        config = json.load(f)

    from src.config import DataConfig
    from src.data.loaders import get_data_loaders
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

    raw = config.get("data_params", {})
    data_cfg = DataConfig(**{
        **{k: v for k, v in raw.items() if k in _DATA_CONFIG_FIELDS},
        "batch_size": 256, "val_batch_size": 256,
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    return model, val_loader


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, val_loader, device, n_batches=50, n_noise_samples=3):
    """
    Returns dict  {sigma_value: {'clean_error': float, 'own_pred_bias': float}}
    evaluated over *n_batches* batches with *n_noise_samples* noise draws each.
    """
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

            sum_clean = None
            sum_own = None
            for _ in range(n_noise_samples):
                _, ests_clean = model(conditioning=cond, data=target,
                                      return_x0_estimate=True, input_type="clean")
                _, ests_own = model(conditioning=cond, data=target,
                                    return_x0_estimate=True, input_type="own-pred")
                batch_clean = torch.stack(
                    [(e - target).pow(2).mean(dim=spatial_dims) for e in ests_clean]
                )  # (T, B)
                batch_own = torch.stack(
                    [(e - target).pow(2).mean(dim=spatial_dims) for e in ests_own]
                )  # (T, B)
                sum_clean = batch_clean if sum_clean is None else sum_clean + batch_clean
                sum_own = batch_own if sum_own is None else sum_own + batch_own

            avg_clean = sum_clean / n_noise_samples
            avg_own = sum_own / n_noise_samples

            for t_idx in range(T):
                clean_i = avg_clean[t_idx]
                own_i = avg_own[t_idx]
                valid = clean_i > 0
                ratio_i = (own_i / clean_i.clamp(min=1e-12))[valid]
                all_clean[t_idx].append(clean_i[valid].cpu())
                all_ratio[t_idx].append(ratio_i.cpu())

    results = {}
    for t_idx in range(T):
        if not all_clean[t_idx]:
            continue
        sigma = float(sigmas[t_idx])
        clean_all = torch.cat(all_clean[t_idx])
        ratio_all = torch.cat(all_ratio[t_idx])
        results[sigma] = {
            "clean_error": float(clean_all.mean()),
            "own_pred_bias": float(ratio_all.mean()),
        }
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

GROUP_STYLES = [
    {"label": r"Full range (1 model)", "color": "#1f77b4", "ls": "-", "marker": "o"},
    {"label": r"Half ranges (2 models)", "color": "#ff7f0e", "ls": "--", "marker": "s"},
    {"label": r"Third ranges (3 models)", "color": "#2ca02c", "ls": "-.", "marker": "^"},
]

# models per group (0-indexed into the 6-element list)
GROUPS = [[0], [1, 2], [3, 4, 5]]


def stitch_results(model_results_list):
    """
    Merge a list of {sigma: {...}} dicts from models in the same group into one
    sorted dict, keeping each sigma's entry from whichever model owns it.
    (No overlap expected; if there is, last model wins.)
    """
    merged = {}
    for r in model_results_list:
        merged.update(r)
    return dict(sorted(merged.items()))


SIGMA_RANGES = [(1e-2, 5e-2), (1e-1, 1e0)]


def plot_comparison(all_results, output_path):
    # 2 rows (metrics) x 2 cols (sigma ranges), shared y within each row
    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                             gridspec_kw={"wspace": 0.08})

    metrics = [
        ("clean_error",   r"Clean-Input Error $\mathcal{E}_{\mathrm{clean}}$"),
        ("own_pred_bias", r"Own-Pred Bias $B^{(\mathrm{own})}$"),
    ]
    col_labels = [r"$\sigma \in [10^{-2},\,10^{-1}]$",
                  r"$\sigma \in [10^{-1},\,10^{0}]$"]

    for group_idx, model_indices in enumerate(GROUPS):
        style = GROUP_STYLES[group_idx]
        merged = stitch_results([all_results[i] for i in model_indices])
        sigmas = np.array(sorted(merged.keys()))
        clean_errors = np.array([merged[s]["clean_error"] for s in sigmas])
        own_biases   = np.array([merged[s]["own_pred_bias"] for s in sigmas])
        values = [clean_errors, own_biases]

        for row, vals in enumerate(values):
            for col, (lo, hi) in enumerate(SIGMA_RANGES):
                mask = (sigmas >= lo) & (sigmas <= hi)
                axes[row, col].plot(
                    sigmas[mask], vals[mask],
                    label=style["label"], color=style["color"],
                    linestyle=style["ls"], marker=style["marker"],
                    markersize=5, zorder=group_idx + 10,
                )

    for row, (_, ylabel) in enumerate(metrics):
        for col, (lo, hi) in enumerate(SIGMA_RANGES):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_xlabel(r"Noise Level $\sigma$", labelpad=6)
            ax.grid(True, which="major", linestyle="-", linewidth=0.75, color="0.85")
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="0.9")
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=8)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(col_labels[col], pad=8)

    axes[1, 0].axhline(1.0, color="gray", linewidth=1.0, linestyle=":", zorder=0)
    axes[1, 1].axhline(1.0, color="gray", linewidth=1.0, linestyle=":", zorder=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True,
               fancybox=False, framealpha=0.95, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

MODEL_SUBDIRS = [
    "model1_full",
    "model2_lower_half",
    "model3_upper_half",
    "model4_lower_third",
    "model5_mid_third",
    "model6_upper_third",
]


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base_dir", help=(
        "Parent directory containing the 6 model subdirs "
        "(model1_full, model2_lower_half, …, model6_upper_third)."
    ))
    group.add_argument("--runs", nargs=6, metavar="RUN_DIR",
                       help="Explicit paths to the 6 run directories, in order: "
                            "model1 model2 model3 model4 model5 model6.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--n_noise_samples", type=int, default=3)
    parser.add_argument("--output", default=None,
                        help="Output PDF path. Defaults to <base_dir>/schedule_comparison.pdf.")
    args = parser.parse_args()

    if args.base_dir:
        parent_dirs = [os.path.join(args.base_dir, s) for s in MODEL_SUBDIRS]
        default_output = os.path.join(args.base_dir, "schedule_comparison.pdf")
    else:
        parent_dirs = args.runs
        default_output = os.path.join(os.path.dirname(args.runs[0]), "schedule_comparison.pdf")

    output_path = args.output or default_output

    # Cache file to avoid re-evaluating if already done
    cache_path = output_path.replace(".pdf", "_cache.json")

    if os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        with open(cache_path) as f:
            raw = json.load(f)
        # JSON keys are strings; convert sigma keys back to float
        all_results = [
            {float(k): v for k, v in entry.items()}
            for entry in raw
        ]
    else:
        all_results = []
        for i, parent in enumerate(parent_dirs):
            run_dir = find_run_dir(parent)
            print(f"[{i+1}/6] Evaluating {run_dir} ...")
            model, val_loader = load_model_and_data(run_dir, args.device)
            results = evaluate_model(model, val_loader, args.device,
                                     n_batches=args.n_batches,
                                     n_noise_samples=args.n_noise_samples)
            all_results.append(results)
            print(f"       {len(results)} sigma levels evaluated.")
            del model  # free GPU memory between models

        with open(cache_path, "w") as f:
            # JSON requires string keys
            json.dump([{str(k): v for k, v in r.items()} for r in all_results], f, indent=2)
        print(f"Cached results to {cache_path}")

    plot_comparison(all_results, output_path)


if __name__ == "__main__":
    main()
