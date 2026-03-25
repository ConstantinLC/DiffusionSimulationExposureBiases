#!/usr/bin/env python
"""
Evaluate any trained model saved in a checkpoint directory on KS (1D) data.
Results are saved inside the checkpoint folder.

Usage (single model):
    python eval_ks.py --checkpoint_dir /path/to/checkpoint

Usage (multiple models):
    python eval_ks.py --checkpoint_dirs /path/a /path/b --output_dir /path/results
"""
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel
from src.models.pderefiner import PDERefiner
from src.models.unet_1d import Unet1D
from src.models.dilresnet import DilatedResNet
from src.models.fno import FNO


# ==========================================
# Utilities
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Seed set to {seed} ---")


def correlation(qa, qb):
    return pearsonr(qa.ravel(), qb.ravel())[0]


def load_config(checkpoint_dir: str) -> dict:
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def build_model_from_legacy(model_params: dict) -> torch.nn.Module:
    """Detect model type from legacy model_params dict and build the model (no checkpoint loaded)."""
    mp = dict(model_params)
    mp["checkpoint"] = ""  # weights loaded separately via state_dict

    if "refinementSteps" in mp:
        return PDERefiner(
            dimension=mp["dimension"],
            dataSize=mp["dataSize"],
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            refinementSteps=mp["refinementSteps"],
            log_sigma_min=mp["log_sigma_min"],
            padding_mode=mp.get("padding_mode", "circular"),
            architecture=mp.get("architecture", "Unet2D"),
            checkpoint="",
        )

    elif "diffSchedule" in mp:
        return DiffusionModel(
            dimension=mp["dimension"],
            dataSize=mp["dataSize"],
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            diffSchedule=mp["diffSchedule"],
            diffSteps=mp["diffSteps"],
            inferenceSamplingMode=mp["inferenceSamplingMode"],
            inferenceConditioningIntegration=mp["inferenceConditioningIntegration"],
            diffCondIntegration=mp["diffCondIntegration"],
            padding_mode=mp.get("padding_mode", "circular"),
            architecture=mp.get("architecture", "Unet2D"),
            checkpoint="",
            load_betas=False,
            schedule_path=mp.get("schedule_path", None),
        )

    elif "blocks" in mp:
        return DilatedResNet(
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            blocks=mp.get("blocks", 4),
            features=mp.get("features", 48),
            dilate=mp.get("dilate", True),
        )

    elif "modes" in mp:
        return FNO(
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            modes=mp["modes"],
            hidden_channels=mp.get("hidden_channels", 64),
            n_layers=mp.get("n_layers", 4),
        )

    else:
        raise ValueError(f"Cannot determine model type from model_params keys: {list(mp.keys())}")


def load_model(checkpoint_dir: str, device: str) -> tuple[torch.nn.Module, str]:
    """Load model from checkpoint_dir; returns (model, model_name)."""
    cfg = load_config(checkpoint_dir)
    model = build_model_from_legacy(cfg["model_params"])

    ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        epoch_ckpts = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        if not epoch_ckpts:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        ckpt_path = os.path.join(checkpoint_dir, epoch_ckpts[-1])
        print(f"  best_model.pth not found, using {epoch_ckpts[-1]}")

    print(f"  Loading weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, os.path.basename(checkpoint_dir)


def run_model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


# ==========================================
# Metrics
# ==========================================

def evaluate_trajectory(predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.8):
    """Pearson correlation per timestep for 1D data. predictions/ground_truth: (N, T, C, L)."""
    N, T, C, L = predictions.shape

    corrs = np.zeros((N, T))
    for n in range(N):
        for t in range(T):
            corrs[n, t] = correlation(predictions[n, t], ground_truth[n, t])

    mean_correlations = np.mean(corrs, axis=0)
    std_per_timestep = np.std(corrs, axis=0)

    below_threshold = corrs < threshold
    first_failure = np.argmax(below_threshold, axis=1)
    has_failed = np.any(below_threshold, axis=1)
    times = np.where(has_failed, first_failure + 1, T)

    sorted_times = np.sort(times)
    cutoff = max(1, int(0.1 * N))

    return {
        "mean_correlations": mean_correlations,
        "std_per_timestep": std_per_timestep,
        "time_under_threshold": float(np.mean(times)),
        "time_under_threshold_worst_10": float(np.mean(sorted_times[:cutoff])),
        "time_under_threshold_best_10": float(np.mean(sorted_times[-cutoff:])),
    }


# ==========================================
# DW Bias Evaluation
# ==========================================

def get_noise_levels(model):
    """Return noise levels low → high (matches MSE array ordering)."""
    if isinstance(model, DiffusionModel):
        # sqrtOneMinusAlphasCumprod is naturally low → high (t=0 first)
        # DiffusionModel applies torch.flip to estimates, so index 0 = lowest noise
        return model.sqrtOneMinusAlphasCumprod.ravel().cpu().tolist()
    elif isinstance(model, PDERefiner):
        # sigmas is low → high; PDERefiner does NOT flip estimates (index 0 = highest sigma)
        # so return reversed to align with estimates order
        return model.sigmas.ravel().cpu().tolist()[::-1]
    return None


def evaluate_dw_bias(model, val_loader, device):
    """
    Run the full validation set through a DiffusionModel or PDERefiner with three
    input_type settings and return per-timestep MSE lists (averaged over all batches,
    ordered low → high noise to match get_noise_levels).
    """
    model.eval()
    n_steps = model.timesteps if isinstance(model, DiffusionModel) else model.nTimesteps
    sum_mse = {k: np.zeros(n_steps) for k in ("ancestor", "clean", "own_pred")}
    n_batches = 0

    # PDERefiner returns estimates in inference order (high → low sigma, index 0 = highest)
    # DiffusionModel torch.flips them (index 0 = lowest noise)
    # Flip PDERefiner estimates so both are low → high (index 0 = lowest noise)
    needs_flip = isinstance(model, PDERefiner)

    with torch.no_grad():
        for sample in val_loader:
            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]

            _, x0_ancestor = model(conditioning=conditioning_frame, data=target_frame,
                                   return_x0_estimate=True, input_type="ancestor")
            _, x0_clean = model(conditioning=conditioning_frame, data=target_frame,
                                return_x0_estimate=True, input_type="clean")
            _, x0_own_pred = model(conditioning=conditioning_frame, data=target_frame,
                                   return_x0_estimate=True, input_type="own-pred")

            if needs_flip:
                x0_ancestor = torch.flip(x0_ancestor, [0])
                x0_clean = torch.flip(x0_clean, [0])
                x0_own_pred = torch.flip(x0_own_pred, [0])

            for t in range(n_steps):
                sum_mse["ancestor"][t]  += torch.mean((x0_ancestor[t]  - target_frame) ** 2).item()
                sum_mse["clean"][t]     += torch.mean((x0_clean[t]     - target_frame) ** 2).item()
                sum_mse["own_pred"][t]  += torch.mean((x0_own_pred[t]  - target_frame) ** 2).item()
            n_batches += 1

    print(f"  DW bias evaluated over {n_batches} batches")
    return {
        "mse_ancestor":       (sum_mse["ancestor"]  / n_batches).tolist(),
        "mse_clean":          (sum_mse["clean"]      / n_batches).tolist(),
        "mse_clean_own_pred": (sum_mse["own_pred"]   / n_batches).tolist(),
    }


def plot_dw_bias(results, noise_levels, name, output_dir):
    # noise_levels are in inference order (high → low); reverse so x-axis goes low → high
    noise_levels = noise_levels
    print(noise_levels)
    print(results)
    results = {k: v if isinstance(v, list) else v for k, v in results.items()}

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    noise_arr = np.array(noise_levels)
    bins = np.logspace(np.log10(max(noise_arr.min(), 1e-10)), np.log10(noise_arr.max()), 20)
    axes[0].hist(noise_levels, bins=bins, alpha=0.4, color='purple')
    axes[0].set_title(f"Noise Distribution: {name}")
    axes[0].set_ylabel('Count')
    axes[0].set_xscale('log')

    axes[1].plot(noise_levels, results["mse_clean"],
                 label="Training input (Clean)", color='blue', linestyle='dotted', linewidth=2)
    axes[1].plot(noise_levels, results["mse_ancestor"],
                 label="Inference input (Ancestor)", color='red', linestyle='solid', linewidth=2)
    axes[1].plot(noise_levels, results["mse_clean_own_pred"],
                 label="Inference input (Own Pred)", color='green', linestyle='dashdot', linewidth=2)

    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].grid(True, which='both', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('MSE w/ Ground Truth')
    axes[1].set_xlabel(r'Noise Level $\sqrt{1-\bar{\alpha}_t}$ (Log Scale)')
    axes[1].legend(fontsize=10)

    summary = (
        f"Final Errors:\n"
        f"  Clean:    {results['mse_clean'][0]:.2e}\n"
        f"  Ancestor: {results['mse_ancestor'][0]:.2e}\n"
        f"  Own-Pred: {results['mse_clean_own_pred'][0]:.2e}"
    )
    axes[1].text(0.05, 0.95, summary, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"dw_bias_{name}.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  DW bias plot saved to {save_path}")


# ==========================================
# Rollout Evaluation
# ==========================================

def evaluate_rollout(models: dict, traj_loader, device: str, rollout_steps: int):
    for model in models.values():
        model.eval()

    all_predictions = {name: [] for name in models}
    all_ground_truth = []

    print(f"Starting evaluation over full dataset ({len(traj_loader)} batches)...")

    with torch.no_grad():
        for batch_idx, sample in enumerate(traj_loader):
            data = sample["data"].to(device)  # (B, T_total, C, L)
            all_ground_truth.append(data.cpu())

            conditioning_frame = data[:, 0]
            current_preds = {name: run_model(model, conditioning_frame) for name, model in models.items()}

            batch_buffers = {name: [current_preds[name]] for name in models}

            for t in range(1, rollout_steps):
                for name, model in models.items():
                    current_preds[name] = run_model(model, current_preds[name])
                    batch_buffers[name].append(current_preds[name])

            for name in models:
                traj = torch.stack(batch_buffers[name], dim=1)  # (B, T, C, L)
                all_predictions[name].append(traj.cpu())

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed batch {batch_idx + 1}/{len(traj_loader)}")

    final_predictions = {name: torch.cat(all_predictions[name], dim=0) for name in models}
    final_ground_truth = torch.cat(all_ground_truth, dim=0)
    return final_predictions, final_ground_truth


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate models from checkpoint directories on KS data")
    parser.add_argument(
        "--checkpoint_dirs", nargs="+", required=True,
        help="One or more checkpoint directories (each must contain config.json and best_model.pth)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results. Defaults to the checkpoint dir when only one is given."
    )
    parser.add_argument("--rollout_steps", type=int, default=159)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.checkpoint_dirs[0] if len(args.checkpoint_dirs) == 1 else "."
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # Load data config from the first checkpoint dir
    first_cfg = load_config(args.checkpoint_dirs[0])
    dp = first_cfg["data_params"]
    data_config = DataConfig(
        dataset_name=dp["dataset_name"],
        data_path=dp["data_path"],
        resolution=dp["resolution"],
        super_resolution=dp.get("super_resolution", False),
        downscale_factor=dp.get("downscale_factor", 4),
        prediction_steps=dp.get("prediction_steps", dp.get("sequence_length", [2, 1])[0] - 1),
        frames_per_step=dp.get("frames_per_step", dp.get("sequence_length", [2, 1])[1]),
        traj_length=dp.get("traj_length", dp.get("trajectory_sequence_length", [160, 1])[0]),
        frames_per_time_step=dp.get("frames_per_time_step", 1),
        limit_trajectories_train=dp.get("limit_trajectories_train", -1),
        limit_trajectories_val=dp.get("limit_trajectories_val", -1),
        batch_size=dp.get("val_batch_size", dp.get("batch_size", 64)),
        val_batch_size=dp.get("val_batch_size", 64),
    )

    print("--- Loading Data ---")
    _, val_loader, traj_loader = get_data_loaders(data_config)

    print("--- Loading Models ---")
    models = {}
    for ckpt_dir in args.checkpoint_dirs:
        model, name = load_model(ckpt_dir, args.device)
        if name in models:
            name = f"{name}_{ckpt_dir}"
        models[name] = model
        print(f"  Loaded: {name}")

    print("--- Running Rollouts ---")
    predictions, ground_truth = evaluate_rollout(models, traj_loader, args.device, args.rollout_steps)

    print("--- Computing Metrics ---")
    gt_trajectory = ground_truth[:, 1:args.rollout_steps + 1]

    fig_mse, ax_mse = plt.subplots(figsize=(10, 6))
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))

    final_metrics = {}

    for name in models:
        preds = predictions[name]
        preds = torch.nan_to_num(preds, nan=1e5, posinf=1e5, neginf=-1e5)
        preds = torch.clamp(preds, -1e5, 1e5)

        # MSE (B, T, C, L) -> mean over B, C, L
        sq_err = torch.clamp((preds - gt_trajectory) ** 2, max=1e10)
        mse_time = np.nan_to_num(torch.mean(sq_err, dim=(0, 2, 3)).cpu().numpy(), nan=1e10, posinf=1e10)
        ax_mse.plot(mse_time, label=name)

        # Pearson correlation
        corr_stats = evaluate_trajectory(preds.numpy(), gt_trajectory.numpy())
        ax_corr.plot(corr_stats["mean_correlations"], label=name)

        def clean(x):
            return float(x) if np.isfinite(x) else 1e10

        final_metrics[name] = {
            "time_to_failure_avg": clean(corr_stats["time_under_threshold"]),
            "time_to_failure_worst10": clean(corr_stats["time_under_threshold_worst_10"]),
            "time_to_failure_best10": clean(corr_stats["time_under_threshold_best_10"]),
            "step1_mse": clean(mse_time[0]),
            "step10_mse": clean(mse_time[10]) if len(mse_time) > 10 else 0.0,
            "last_step_mse": clean(mse_time[-1]),
        }

    # Finalize and save plots
    ax_mse.set_yscale("log")
    ax_mse.set_title("MSE vs Time Step")
    ax_mse.set_xlabel("Time Step")
    ax_mse.set_ylabel("MSE")
    ax_mse.legend()
    fig_mse.savefig(os.path.join(args.output_dir, "metric_mse.png"))
    plt.close(fig_mse)

    ax_corr.set_title("Pearson Correlation vs Time Step")
    ax_corr.set_xlabel("Time Step")
    ax_corr.set_ylabel("Pearson Correlation")
    ax_corr.set_ylim(0, 1.05)
    ax_corr.axhline(0.8, color="black", linestyle="--", alpha=0.5, label="Failure Threshold")
    ax_corr.legend()
    fig_corr.savefig(os.path.join(args.output_dir, "metric_correlation.png"))
    plt.close(fig_corr)

    metrics_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"Done. Results saved to {args.output_dir}")
    for name, m in final_metrics.items():
        print(f"\n  {name}:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    # --- DW Bias Evaluation (DiffusionModel only) ---
    print("\n--- DW Bias Evaluation ---")
    all_bias_metrics = {}
    for name, model in models.items():
        if not isinstance(model, (DiffusionModel, PDERefiner)):
            print(f"  Skipping {name} (not a DiffusionModel or PDERefiner)")
            continue
        print(f"  Evaluating {name}...")
        bias_results = evaluate_dw_bias(model, val_loader, args.device)
        noise_levels = get_noise_levels(model)

        plot_dw_bias(bias_results, noise_levels, name, args.output_dir)

        all_bias_metrics[name] = {
            "noise_levels": noise_levels,
            "mse_ancestor": bias_results["mse_ancestor"],
            "mse_clean": bias_results["mse_clean"],
            "mse_clean_own_pred": bias_results["mse_clean_own_pred"],
            "final_mse_ancestor": bias_results["mse_ancestor"][0],
            "final_mse_clean": bias_results["mse_clean"][0],
            "final_mse_clean_own_pred": bias_results["mse_clean_own_pred"][0],
        }
        print(f"    final mse_ancestor:  {bias_results['mse_ancestor'][0]:.4e}")
        print(f"    final mse_clean:     {bias_results['mse_clean'][0]:.4e}")
        print(f"    final mse_own_pred:  {bias_results['mse_clean_own_pred'][0]:.4e}")

    if all_bias_metrics:
        bias_path = os.path.join(args.output_dir, "dw_bias_metrics.json")
        with open(bias_path, "w") as f:
            json.dump(all_bias_metrics, f, indent=4)
        print(f"  DW bias metrics saved to {bias_path}")


if __name__ == "__main__":
    main()
