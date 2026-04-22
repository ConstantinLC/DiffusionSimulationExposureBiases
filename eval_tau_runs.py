"""
Evaluate greedy-trained models on the KS validation set.

Original usage (hardcoded runs 22, 23, 24):
    python eval_tau_runs.py [--device cuda] [--output_dir results/]

Multi-seed usage (from run_tau_seeds.sh manifest):
    python eval_tau_runs.py --manifest tau_seeds_manifest.json \\
        [--device cuda] [--output_dir results/tau_seeds/]

Directory usage (auto-discover tau_<V>/run_<K>/ layout):
    python eval_tau_runs.py --runs_dir checkpoints/KS/tau_grid \\
        [--device cuda] [--output_dir results/tau_seeds/]

In manifest/runs_dir mode the script evaluates all seeds per tau, computes
per-seed DW-bias metrics, then averages across seeds (via interpolation onto
a shared log-spaced grid) and plots mean ± std for each tau.
"""

import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.eval_ks import (
    load_model,
    load_config,
    evaluate_dw_bias,
    get_noise_levels,
)
from src.config import DataConfig
from src.data.loaders import get_data_loaders

BASE = "/mnt/SSD2/constantin/diffusion-multisteps/checkpoints/KuramotoSivashinsky/exploration"

RUNS = {
    "run_22": {"tau": 1.10, "color": "#e41a1c",
               "ckpt_dir": f"{BASE}/run_22/greedy_trained"},
    "run_23": {"tau": 1.03, "color": "#377eb8",
               "ckpt_dir": f"{BASE}/run_23/greedy_trained"},
    "run_24": {"tau": 1.05, "color": "#4daf4a",
               "ckpt_dir": f"{BASE}/run_24/greedy_trained"},
    "run_25": {"tau": 1.01, "color": "#d8c42f",
               "ckpt_dir": f"{BASE}/run_25/greedy_trained"},
}

TAU_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]


def manifest_from_runs_dir(runs_dir):
    """Build manifest from <runs_dir>/tau_<V>/run_<K>/greedy_trained/ layout."""
    manifest = {}
    for entry in sorted(os.listdir(runs_dir)):
        if not entry.startswith("tau_"):
            continue
        tau_str = entry[len("tau_"):]
        tau_dir = os.path.join(runs_dir, entry)
        runs = []
        for r in sorted(os.listdir(tau_dir)):
            run_dir = os.path.join(tau_dir, r)
            if not os.path.isdir(run_dir):
                continue
            if not os.path.isdir(os.path.join(run_dir, "greedy_schedule")):
                continue
            greedy_trained = os.path.join(run_dir, "greedy_trained")
            ckpt = greedy_trained if os.path.isdir(greedy_trained) else run_dir
            runs.append(ckpt)
        if runs:
            manifest[tau_str] = runs
    return manifest


# ── data loader ────────────────────────────────────────────────────────────────

def build_val_loader(ckpt_dir: str, device: str):
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


# ── averaging helpers ──────────────────────────────────────────────────────────

def interp_to_grid(noise_levels, values, grid):
    """Log-interpolate values onto a common grid (both sorted ascending)."""
    log_nl = np.log10(np.maximum(noise_levels, 1e-30))
    log_grid = np.log10(np.maximum(grid, 1e-30))
    return np.interp(log_grid, log_nl, values)


def average_seeds(seed_results, grid):
    """
    seed_results: list of {"noise_levels": [...], "bias": {...}}
    Returns mean and std of mse_clean and mse_clean_own_pred on grid.
    """
    keys = ["mse_clean", "mse_clean_own_pred"]
    interped = {k: [] for k in keys}
    for res in seed_results:
        nl = np.array(res["noise_levels"])
        for k in keys:
            interped[k].append(interp_to_grid(nl, np.array(res["bias"][k]), grid))
    return {
        k: {
            "mean": np.mean(interped[k], axis=0),
            "std":  np.std(interped[k],  axis=0),
            "all":  interped[k],
        }
        for k in keys
    }


# ── original single-run plot (unchanged behaviour) ────────────────────────────

def plot_single_runs(all_results, output_dir):
    fig, ax_err = plt.subplots(figsize=(8, 5))
    fig.suptitle("Greedy-trained models: noise level vs one-step error (validation)", fontsize=13)

    records = {}
    for run_name, res in all_results.items():
        nl    = np.array(res["noise_levels"])
        own   = np.array(res["bias"]["mse_clean_own_pred"])
        c     = res["color"]
        tau   = res["tau"]

        ax_err.semilogy(nl, own, color=c, marker="s", markersize=5,
                        linewidth=1.8, linestyle="-",
                        label=f"{run_name} (τ={tau})")
        records[run_name] = {
            "tau": tau,
            "noise_levels": nl.tolist(),
            "mse_clean": res["bias"]["mse_clean"],
            "mse_inference": res["bias"]["mse_clean_own_pred"],
        }

    ax_err.set_xscale("log")
    ax_err.set_xlabel(r"Noise level $\sigma$")
    ax_err.set_ylabel("MSE (inference)")
    ax_err.set_title("One-step inference error")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(output_dir, "tau_runs_noise_vs_error.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Plot saved → {out}")
    out_png = os.path.join(output_dir, "tau_runs_noise_vs_error.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Plot saved → {out_png}")
    plt.show()

    data_out = os.path.join(output_dir, "tau_runs_errors.json")
    with open(data_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Data saved  → {data_out}")


# ── multi-seed averaged plot ───────────────────────────────────────────────────

def plot_averaged(tau_averaged, output_dir):
    """
    tau_averaged: {tau_str: {"grid": [...], "averaged": {...}, "color": ...,
                              "tau": float, "seed_results": [...]}}
    """
    fig, ax_err = plt.subplots(figsize=(5, 5))

    records = {}
    for tau_str, info in tau_averaged.items():
        grid    = np.array(info["grid"])
        avg     = info["averaged"]
        c       = info["color"]
        tau     = info["tau"]
        n_seeds = len(info["seed_results"])

        scores = [r["bias"]["mse_clean_own_pred"][0] for r in info["seed_results"]]
        order = np.argsort(scores)
        pick = order[1] if len(order) > 1 else order[0]
        seed_res = info["seed_results"][pick]
        nl_seed = np.array(seed_res["noise_levels"])
        seed_own = np.array(seed_res["bias"]["mse_clean_own_pred"])
        val0 = seed_own[0]
        ax_err.semilogy(nl_seed, seed_own, color=c, linewidth=2.0,
                        linestyle="-", alpha=0.9, label=f"τ={tau**0.5:.4f}  [{val0:.2e}]")

        seed_results = info["seed_results"]
        nl0      = np.array([r["noise_levels"][0]                    for r in seed_results])
        clean0   = np.array([r["bias"]["mse_clean"][0]               for r in seed_results])
        infer0   = np.array([r["bias"]["mse_clean_own_pred"][0]      for r in seed_results])
        ratio0   = np.sqrt(infer0 / clean0)
        ratio0_mean = float(np.mean(ratio0))
        print(f"  sqrt(mse_inf[0]/mse_clean[0]) per run: {ratio0.tolist()}")
        print(f"  mean sqrt(mse_inf[0]/mse_clean[0]): {ratio0_mean:.6f}")
        records[tau_str] = {
            "tau": tau,
            "noise_levels":      float(np.mean(nl0)),
            "mse_clean_mean":    float(np.mean(clean0)),
            "mse_clean_std":     float(np.std(clean0)),
            "mse_inference_mean": float(np.mean(infer0)),
            "mse_inference_std":  float(np.std(infer0)),
            "sqrt_mse_inf_over_clean_mean": ratio0_mean,
            "sqrt_mse_inf_over_clean_per_run": ratio0.tolist(),
            "runs": [
                {
                    "ckpt_dir": r["ckpt_dir"],
                    "noise_levels": r["noise_levels"],
                    "mse_clean": r["bias"]["mse_clean"],
                    "mse_inference": r["bias"]["mse_clean_own_pred"],
                }
                for r in info["seed_results"]
            ],
        }

    ax_err.set_xscale("log")
    ax_err.set_xlabel(r"Noise level $\sigma$", fontsize=14)
    ax_err.set_ylabel("MSE (inference)", fontsize=14)
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(fontsize=12)

    plt.tight_layout()
    out = os.path.join(output_dir, "tau_seeds_noise_vs_error.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Plot saved  → {out}")
    out_png = os.path.join(output_dir, "tau_seeds_noise_vs_error.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Plot saved  → {out_png}")
    plt.show()

    data_out = os.path.join(output_dir, "tau_seeds_errors.json")
    with open(data_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Data saved  → {data_out}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--manifest", default=None,
        help="Path to tau_seeds_manifest.json produced by run_tau_seeds.sh. "
             "When provided, evaluates all seeds per tau and averages results.",
    )
    parser.add_argument(
        "--runs_dir", default=None,
        help="Directory with tau_<V>/run_<K>/ layout; auto-builds manifest.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── manifest / runs_dir mode ───────────────────────────────────────────────
    if args.runs_dir or args.manifest:
        if args.runs_dir:
            manifest = manifest_from_runs_dir(args.runs_dir)
            print(f"Discovered runs_dir: {args.runs_dir}")
        else:
            with open(args.manifest) as f:
                manifest = json.load(f)  # {tau_str: [ckpt_dir, ...]}

        print(f"Manifest loaded: {len(manifest)} tau value(s)")

        # Build val loader from the first checkpoint
        first_ckpt = next(iter(manifest.values()))[0]
        print("--- Loading data ---")
        val_loader = build_val_loader(first_ckpt, args.device)

        tau_averaged = {}
        all_per_seed_metrics = {}

        for idx, (tau_str, ckpt_dirs) in enumerate(sorted(manifest.items(),
                                                           key=lambda x: float(x[0]))):
            tau = float(tau_str)
            color = TAU_COLORS[idx % len(TAU_COLORS)]
            print(f"\n=== tau={tau}  ({len(ckpt_dirs)} seed(s)) ===")

            seed_results = []
            for seed_idx, ckpt_dir in enumerate(ckpt_dirs):
                print(f"  -- seed {seed_idx + 1}: {ckpt_dir}")
                model, _ = load_model(ckpt_dir, args.device)
                bias = evaluate_dw_bias(model, val_loader, args.device)
                noise_levels = get_noise_levels(model)
                seed_results.append({"noise_levels": noise_levels, "bias": bias,
                                     "ckpt_dir": ckpt_dir})
                del model
                torch.cuda.empty_cache()

            # Use first seed's actual noise levels as the interpolation grid
            grid = np.array(seed_results[0]["noise_levels"])

            averaged = average_seeds(seed_results, grid)

            tau_averaged[tau_str] = {
                "tau": tau, "color": color,
                "grid": grid.tolist(),
                "averaged": averaged,
                "seed_results": seed_results,
            }
            all_per_seed_metrics[tau_str] = {
                "tau": tau,
                "seeds": [
                    {"ckpt_dir": r["ckpt_dir"],
                     "noise_levels": r["noise_levels"],
                     **r["bias"]}
                    for r in seed_results
                ],
                "averaged": {
                    k: {
                        "grid": grid.tolist(),
                        "mean": averaged[k]["mean"].tolist(),
                        "std":  averaged[k]["std"].tolist(),
                    }
                    for k in ("mse_clean", "mse_clean_own_pred")
                },
            }

        plot_averaged(tau_averaged, args.output_dir)
        return

    # ── original hardcoded mode ────────────────────────────────────────────────
    first_cfg = load_config(next(iter(RUNS.values()))["ckpt_dir"])
    dp = first_cfg["data_params"]
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
    print("--- Loading data ---")
    _, val_loader, _ = get_data_loaders(data_config)

    all_results = {}
    for run_name, meta in RUNS.items():
        print(f"\n--- {run_name}  (τ={meta['tau']}) ---")
        model, _ = load_model(meta["ckpt_dir"], args.device)
        bias = evaluate_dw_bias(model, val_loader, args.device)
        noise_levels = get_noise_levels(model)
        all_results[run_name] = {"bias": bias, "noise_levels": noise_levels, **meta}
        del model
        torch.cuda.empty_cache()

    plot_single_runs(all_results, args.output_dir)


if __name__ == "__main__":
    main()
