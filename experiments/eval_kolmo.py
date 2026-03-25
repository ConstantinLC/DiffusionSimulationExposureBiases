#!/usr/bin/env python
"""
Evaluate any trained model saved in a checkpoint directory.
Results are saved inside the checkpoint folder.

Usage (single model):
    python eval_kolmo.py --checkpoint_dir /path/to/checkpoint

Usage (multiple models, saved to first dir):
    python eval_kolmo.py --checkpoint_dirs /path/a /path/b --output_dir /path/results
"""
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig, DiffusionModelConfig, RefinerConfig, Unet2DConfig, DilResNetConfig, FNOConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel
from src.models.pderefiner import PDERefiner
from src.models.unet_2d import Unet
from src.models.dilresnet import DilatedResNet
from src.models.fno import FNO
from src.utils.general import vorticity, fsd_torch_radial


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
    model_params = cfg["model_params"]
    model = build_model_from_legacy(model_params)

    ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        # Fall back to latest epoch checkpoint
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

    name = os.path.basename(checkpoint_dir)
    return model, name


def run_model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a single forward inference step."""
    with torch.no_grad():
        return model(x)


# ==========================================
# Metrics
# ==========================================

def evaluate_trajectory_vorticity(predictions: torch.Tensor, ground_truth: torch.Tensor, threshold: float = 0.8):
    predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    N, T, C, H, W = predictions.shape

    pred_flat = predictions.reshape(-1, C, H, W)
    gt_flat = ground_truth.reshape(-1, C, H, W)

    pred_vort = vorticity(pred_flat).reshape(N, T, H, W).cpu().numpy()
    gt_vort = vorticity(gt_flat).reshape(N, T, H, W).cpu().numpy()

    pred_s = pred_vort.reshape(N, T, -1)
    gt_s = gt_vort.reshape(N, T, -1)

    pred_mean = pred_s - pred_s.mean(axis=2, keepdims=True)
    gt_mean = gt_s - gt_s.mean(axis=2, keepdims=True)

    numerator = np.sum(pred_mean * gt_mean, axis=2)
    pred_sq = np.sum(pred_mean ** 2, axis=2)
    gt_sq = np.sum(gt_mean ** 2, axis=2)
    denominator = np.sqrt(pred_sq * gt_sq)

    eps = 1e-8
    corrs = numerator / (denominator + eps)
    corrs[denominator < eps] = 0.0

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
            data = sample["data"].to(device)  # (B, T_total, C, H, W)
            all_ground_truth.append(data.cpu())

            conditioning_frame = data[:, 0]
            current_preds = {name: run_model(model, conditioning_frame) for name, model in models.items()}

            batch_buffers = {name: [current_preds[name]] for name in models}

            for t in range(1, rollout_steps):
                for name, model in models.items():
                    current_preds[name] = run_model(model, current_preds[name])
                    batch_buffers[name].append(current_preds[name])

            for name in models:
                traj = torch.stack(batch_buffers[name], dim=1)  # (B, T, C, H, W)
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
    parser = argparse.ArgumentParser(description="Evaluate models from checkpoint directories")
    parser.add_argument(
        "--checkpoint_dirs", nargs="+", required=True,
        help="One or more checkpoint directories (each must contain config.json and best_model.pth)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results. Defaults to the checkpoint dir when only one is given."
    )
    parser.add_argument("--rollout_steps", type=int, default=63)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        if len(args.checkpoint_dirs) == 1:
            args.output_dir = args.checkpoint_dirs[0]
        else:
            args.output_dir = "."
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # Load data using config from the first checkpoint dir
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
        traj_length=dp.get("traj_length", dp.get("trajectory_sequence_length", [64, 1])[0]),
        frames_per_time_step=dp.get("frames_per_time_step", 1),
        limit_trajectories_train=dp.get("limit_trajectories_train", -1),
        limit_trajectories_val=dp.get("limit_trajectories_val", -1),
        batch_size=dp.get("val_batch_size", dp.get("batch_size", 32)),
        val_batch_size=dp.get("val_batch_size", 32),
    )

    print("--- Loading Data ---")
    _, _, traj_loader = get_data_loaders(data_config)

    print("--- Loading Models ---")
    models = {}
    for ckpt_dir in args.checkpoint_dirs:
        model, name = load_model(ckpt_dir, args.device)
        # Use dir name; append index if duplicate
        if name in models:
            name = f"{name}_{ckpt_dir}"
        models[name] = model
        print(f"  Loaded: {name}")

    print("--- Running Rollouts ---")
    predictions, ground_truth = evaluate_rollout(models, traj_loader, args.device, args.rollout_steps)

    print("--- Computing Metrics ---")
    gt_trajectory = ground_truth[:, 1:args.rollout_steps + 1]

    fig_mse, ax_mse = plt.subplots(figsize=(10, 6))
    fig_fsd, ax_fsd = plt.subplots(figsize=(10, 6))
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))

    final_metrics = {}

    for name in models:
        preds = predictions[name]
        preds = torch.nan_to_num(preds, nan=1e5, posinf=1e5, neginf=-1e5)
        preds = torch.clamp(preds, -1e5, 1e5)

        # MSE
        sq_err = torch.clamp((preds - gt_trajectory) ** 2, max=1e10)
        mse_time = np.nan_to_num(torch.mean(sq_err, dim=(0, 2, 3, 4)).cpu().numpy(), nan=1e10, posinf=1e10)
        ax_mse.plot(mse_time, label=name)

        # FSD
        fsd_time = []
        for t in range(args.rollout_steps):
            try:
                val = fsd_torch_radial(preds[:, t], gt_trajectory[:, t]).item()
                if not np.isfinite(val):
                    val = 1e5
            except Exception:
                val = 1e5
            fsd_time.append(val)
        ax_fsd.plot(fsd_time, label=name)

        # Vorticity correlation
        vort_stats = evaluate_trajectory_vorticity(preds, gt_trajectory)
        ax_corr.plot(vort_stats["mean_correlations"], label=name)

        def clean(x):
            return float(x) if np.isfinite(x) else 1e10

        final_metrics[name] = {
            "time_to_failure_avg": clean(vort_stats["time_under_threshold"]),
            "time_to_failure_worst10": clean(vort_stats["time_under_threshold_worst_10"]),
            "time_to_failure_best10": clean(vort_stats["time_under_threshold_best_10"]),
            "final_fsd": clean(fsd_time[-1]),
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

    ax_fsd.set_yscale("log")
    ax_fsd.set_title("FSD (Radial Spectrum) vs Time Step")
    ax_fsd.set_xlabel("Time Step")
    ax_fsd.set_ylabel("FSD")
    ax_fsd.legend()
    fig_fsd.savefig(os.path.join(args.output_dir, "metric_fsd.png"))
    plt.close(fig_fsd)

    ax_corr.set_title("Vorticity Correlation vs Time Step")
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


if __name__ == "__main__":
    main()
