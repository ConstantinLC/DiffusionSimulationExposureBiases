"""
Evaluate greedy-trained models across different model sizes (dataSize) on the
KS validation set and plot inference MSE vs noise level.

Usage:
    python eval_model_size_runs.py --manifest model_size_manifest.json \\
        [--device cuda] [--output_dir results/model_size/]
    python eval_model_size_runs.py --runs_dir checkpoints/KS/model_size \\
        [--device cuda] [--output_dir results/model_size/]
"""

import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.eval_ks import load_model, load_config, evaluate_dw_bias, get_noise_levels
from src.config import DataConfig
from src.data.loaders import get_data_loaders

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]


def build_val_loader(ckpt_dir):
    cfg = load_config(ckpt_dir)
    dp = cfg["data_params"]
    data_config = DataConfig(
        dataset_name=dp["dataset_name"],
        data_path=dp["data_path"],
        resolution=dp["resolution"],
        super_resolution=dp.get("super_resolution", False),
        downscale_factor=dp.get("downscale_factor", 4),
        prediction_steps=dp.get("prediction_steps",
                                dp.get("sequence_length", [2, 1])[0] - 1),
        frames_per_step=dp.get("frames_per_step",
                               dp.get("sequence_length", [2, 1])[1]),
        traj_length=dp.get("traj_length",
                           dp.get("trajectory_sequence_length", [160, 1])[0]),
        frames_per_time_step=dp.get("frames_per_time_step", 1),
        limit_trajectories_train=dp.get("limit_trajectories_train", -1),
        limit_trajectories_val=dp.get("limit_trajectories_val", -1),
        batch_size=dp.get("val_batch_size", dp.get("batch_size", 64)),
        val_batch_size=dp.get("val_batch_size", 64),
    )
    _, val_loader, _ = get_data_loaders(data_config)
    return val_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def manifest_from_runs_dir(runs_dir):
    """Build manifest from <runs_dir>/size_<N>/run_<K>/greedy_trained/ layout."""
    manifest = {}
    for entry in sorted(os.listdir(runs_dir)):
        if not entry.startswith("size_"):
            continue
        size_str = entry[len("size_"):]
        size_dir = os.path.join(runs_dir, entry)
        runs = []
        for r in sorted(os.listdir(size_dir)):
            run_dir = os.path.join(size_dir, r)
            if not os.path.isdir(run_dir):
                continue
            greedy_trained = os.path.join(run_dir, "greedy_trained")
            ckpt = greedy_trained if os.path.isdir(greedy_trained) else run_dir
            runs.append(ckpt)
        if runs:
            manifest[size_str] = runs
    return manifest


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest",
                       help="model_size_manifest.json from run_model_size.sh")
    group.add_argument("--runs_dir",
                       help="Directory with size_<N>/run_<K>/ layout")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--output_dir", default="results/model_size")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.runs_dir:
        manifest = manifest_from_runs_dir(args.runs_dir)
        print(f"Discovered runs_dir: {args.runs_dir}")
    else:
        with open(args.manifest) as f:
            manifest = json.load(f)  # {dataSize_str: [ckpt_dir, ...]}

    sizes = sorted(manifest.keys(), key=lambda x: int(x))
    print(f"Manifest loaded: {len(sizes)} model size(s): {sizes}")

    first_ckpt = manifest[sizes[0]][0]
    print("--- Loading data ---")
    val_loader = build_val_loader(first_ckpt)

    records = {}
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, size_str in enumerate(sizes):
        ckpt_dirs = manifest[size_str]
        color = COLORS[idx % len(COLORS)]
        n_seeds = len(ckpt_dirs)
        print(f"\n=== dataSize={size_str}  ({n_seeds} seed(s)) ===")

        seed_nls, seed_owns, seed_cleans = [], [], []
        n_params = None
        for seed_idx, ckpt_dir in enumerate(ckpt_dirs):
            print(f"  -- seed {seed_idx + 1}: {ckpt_dir}")
            model, _ = load_model(ckpt_dir, args.device)
            if n_params is None:
                n_params = count_parameters(model)
            bias = evaluate_dw_bias(model, val_loader, args.device)
            noise_levels = get_noise_levels(model)
            del model
            torch.cuda.empty_cache()
            seed_nls.append(np.array(noise_levels))
            seed_owns.append(np.array(bias["mse_clean_own_pred"]))
            seed_cleans.append(np.array(bias["mse_clean"]))

        print(f"  dataSize={size_str}  n_params={n_params:,}")

        # Use first seed's noise levels as the grid; interpolate other seeds onto it
        grid = seed_nls[0]

        def interp(nl, vals):
            return np.interp(np.log10(grid), np.log10(np.maximum(nl, 1e-30)), vals)

        own_interped   = [interp(nl, v) for nl, v in zip(seed_nls, seed_owns)]
        clean_interped = [interp(nl, v) for nl, v in zip(seed_nls, seed_cleans)]
        own_mean = np.mean(own_interped, axis=0)
        own_std  = np.std(own_interped,  axis=0)

        # individual seed curves
        for seed_own in own_interped:
            ax.semilogy(grid, seed_own, color=color, linewidth=0.8, alpha=0.25)

        # mean ± std
        ax.semilogy(grid, own_mean, color=color, linewidth=2.0,
                    label=f"dataSize={size_str}  ({n_params:,} params, n={n_seeds})")
        ax.fill_between(grid,
                        np.maximum(own_mean - own_std, 1e-30),
                        own_mean + own_std,
                        color=color, alpha=0.15)

        records[size_str] = {
            "dataSize": int(size_str),
            "n_params": n_params,
            "n_seeds": n_seeds,
            "noise_levels": grid.tolist(),
            "mse_clean_mean":     np.mean(clean_interped, axis=0).tolist(),
            "mse_clean_std":      np.std(clean_interped,  axis=0).tolist(),
            "mse_inference_mean": own_mean.tolist(),
            "mse_inference_std":  own_std.tolist(),
        }

    ax.set_xscale("log")
    ax.set_xlabel(r"Noise level $\sigma$")
    ax.set_ylabel("MSE (inference)")
    ax.set_title(f"One-step inference error vs model size  (τ=1.05, shading = ±1 std)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    plot_out = os.path.join(args.output_dir, "model_size_noise_vs_error.pdf")
    plt.savefig(plot_out, bbox_inches="tight")
    print(f"\nPlot saved  → {plot_out}")
    plt.show()

    data_out = os.path.join(args.output_dir, "model_size_errors.json")
    with open(data_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Data saved  → {data_out}")


if __name__ == "__main__":
    main()
