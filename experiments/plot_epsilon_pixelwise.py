#!/usr/bin/env python
"""
Visualize epsilon_clean vs epsilon_own_pred pixelwise for different samples.

For a few validation samples, plots the spatial (pixel-wise) x0 predictions
at each noise level for input_type='clean' vs input_type='own-pred',
alongside the ground truth target.

Usage:
    python plot_epsilon_pixelwise.py --checkpoint_dir /path/to/checkpoint
    python plot_epsilon_pixelwise.py --checkpoint_dir /path/to/checkpoint --n_samples 4 --output_dir /path/out
"""
import os
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel
from src.models.pderefiner import PDERefiner
from src.models.unet_1d import Unet1D
from src.models.dilresnet import DilatedResNet
from src.models.fno import FNO


# ==========================================
# Loading utilities (kept from eval_ks.py)
# ==========================================

def load_config(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        return json.load(f)


def build_model_from_legacy(model_params: dict) -> torch.nn.Module:
    mp = dict(model_params)
    mp["checkpoint"] = ""

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
        raise ValueError(f"Cannot determine model type from keys: {list(mp.keys())}")


def load_model(checkpoint_dir: str, device: str, model_file: str = None):
    cfg = load_config(checkpoint_dir)
    model = build_model_from_legacy(cfg["model_params"])

    if model_file is not None:
        ckpt_path = os.path.join(checkpoint_dir, model_file)
    else:
        ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            epoch_ckpts = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")],
                key=lambda x: int(x.split("_")[1].split(".")[0])
            )
            if not epoch_ckpts:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
            ckpt_path = os.path.join(checkpoint_dir, epoch_ckpts[-1])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg, os.path.basename(ckpt_path)


def get_noise_levels(model):
    """Return noise levels low → high."""
    if isinstance(model, DiffusionModel):
        return model.sqrtOneMinusAlphasCumprod.ravel().cpu().numpy()
    elif isinstance(model, PDERefiner):
        return model.sigmas.ravel().cpu().numpy()[::-1]
    return None


# ==========================================
# Core: extract x0 predictions for one sample
# ==========================================

def get_x0_predictions(model, conditioning_frame, target_frame, device):
    """
    Pass 1: clean pass on the original target → x0_clean (T, C, L).
    Pass 2: clean pass using x0_clean[0] (lowest noise prediction) as data → x0_own_pred.
    Ordered low → high noise (index 0 = lowest noise level).
    """
    needs_flip = isinstance(model, PDERefiner)

    with torch.no_grad():
        cond = conditioning_frame.unsqueeze(0).to(device)  # (1, C, L)
        tgt  = target_frame.unsqueeze(0).to(device)         # (1, C, L)

        # Pass 1: clean input
        _, x0_clean = model(conditioning=cond, data=tgt,
                            return_x0_estimate=True, input_type="clean")

        if needs_flip:
            x0_clean = torch.flip(x0_clean, [0])

        # Pass 2: use x0_clean[0] (lowest noise) as data, clean pass
        x0_pred_input = x0_clean[0, 0]  # (C, L)
        _, x0_own_pred = model(conditioning=cond, data=x0_pred_input.unsqueeze(0),
                               return_x0_estimate=True, input_type="clean")

        if needs_flip:
            x0_own_pred = torch.flip(x0_own_pred, [0])

    # squeeze batch dim → (T, C, L)
    x0_clean    = x0_clean[:, 0].cpu().numpy()
    x0_own_pred = x0_own_pred[:, 0].cpu().numpy()
    target_np   = target_frame.cpu().numpy()  # (C, L)

    return x0_clean, x0_own_pred, target_np



# ==========================================
# Main
# ==========================================

def fill_sample_ax(ax, avg_err_clean, avg_err_own_pred, s_idx, label, with_legend):
    mean_clean    = float(avg_err_clean.mean())
    mean_own_pred = float(avg_err_own_pred.mean())
    x = np.arange(len(avg_err_clean))
    ax.plot(x, avg_err_clean,    color="blue", linewidth=1.0, label="(x0_clean - target)²")
    ax.plot(x, avg_err_own_pred, color="red",  linewidth=1.0, label="(x0_own_pred - target)²")
    ax.set_title(f"sample {s_idx}  ({label})", fontsize=9)
    ax.set_xlabel(f"clean={mean_clean:.3e}  own_pred={mean_own_pred:.3e}", fontsize=7)
    if with_legend:
        ax.legend(fontsize=6)


def fill_scatter_ax(ax, scatter_x, scatter_y, label):
    ax.scatter(scatter_x, scatter_y, color="purple", s=20, alpha=0.6, zorder=3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("mean clean error", fontsize=8)
    ax.set_ylabel("own_pred / clean error", fontsize=8)
    ax.set_title(f"error amplification  ({label})", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of validation samples to visualise")
    parser.add_argument("--n_noise_samples", type=int, default=10,
                        help="Number of Gaussian noise realizations per data sample (rows)")
    parser.add_argument("--n_scatter_samples", type=int, default=200,
                        help="Number of data samples to use for the scatter plot")
    parser.add_argument("--model_file", type=str, default='best_model.pth',
                        help="Checkpoint filename inside checkpoint_dir (e.g. best_model.pth, epoch_10.pth). "
                             "Defaults to best_model.pth or the latest epoch checkpoint.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model, cfg, model_name = load_model(args.checkpoint_dir, args.device, args.model_file)

    if not isinstance(model, (DiffusionModel, PDERefiner)):
        raise ValueError("This script requires a DiffusionModel or PDERefiner checkpoint.")

    noise_levels = get_noise_levels(model)

    # Load data
    print("Loading data...")
    dp = cfg["data_params"]
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
    _, val_loader, _ = get_data_loaders(data_config)

    noise_indices = [0, 1, 2]
    labels        = [f"noise={noise_levels[i]:.4f}" for i in noise_indices]

    # Collect samples
    samples = []
    for batch in val_loader:
        data = batch["data"]  # (B, T, C, L)
        for i in range(data.shape[0]):
            samples.append((data[i, 0], data[i, 1]))
            if len(samples) >= args.n_samples:
                break
        if len(samples) >= args.n_samples:
            break

    n_rows = len(noise_indices)
    n_cols = args.n_samples + 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f"Squared errors averaged over {args.n_noise_samples} noise draws  |  {model_name}", fontsize=11)

    # --- Per-sample plots (one pass, accumulate for all noise levels) ---
    print(f"Plotting {len(samples)} samples, averaging over {args.n_noise_samples} noise draws...")
    for s_idx, (cond_frame, tgt_frame) in enumerate(samples):
        sum_err_clean    = {ni: None for ni in noise_indices}
        sum_err_own_pred = {ni: None for ni in noise_indices}

        for _ in range(args.n_noise_samples):
            x0_clean, x0_own_pred, target_np_ = get_x0_predictions(
                model, cond_frame, tgt_frame, args.device
            )
            target_1d = target_np_[0]
            for ni in noise_indices:
                ec = (x0_clean[ni, 0]    - target_1d) ** 2
                eo = (x0_own_pred[ni, 0] - target_1d) ** 2
                sum_err_clean[ni]    = ec if sum_err_clean[ni]    is None else sum_err_clean[ni]    + ec
                sum_err_own_pred[ni] = eo if sum_err_own_pred[ni] is None else sum_err_own_pred[ni] + eo

        for k, ni in enumerate(noise_indices):
            avg_c = sum_err_clean[ni]    / args.n_noise_samples
            avg_o = sum_err_own_pred[ni] / args.n_noise_samples
            fill_sample_ax(axes[k, s_idx], avg_c, avg_o,
                           s_idx, labels[k], with_legend=(s_idx == 0))

    # --- Scatter plots (one pass, accumulate for all noise levels) ---
    print(f"Collecting scatter data from {args.n_scatter_samples} samples...")
    scatter = {ni: ([], []) for ni in noise_indices}
    n_collected = 0
    for batch in val_loader:
        data = batch["data"]
        for i in range(data.shape[0]):
            if n_collected >= args.n_scatter_samples:
                break
            cond_frame_s, tgt_frame_s = data[i, 0], data[i, 1]
            sum_c = {ni: None for ni in noise_indices}
            sum_o = {ni: None for ni in noise_indices}
            for _ in range(args.n_noise_samples):
                xc, xo, tnp = get_x0_predictions(model, cond_frame_s, tgt_frame_s, args.device)
                t1d = tnp[0]
                for ni in noise_indices:
                    ec = (xc[ni, 0] - t1d) ** 2
                    eo = (xo[ni, 0] - t1d) ** 2
                    sum_c[ni] = ec if sum_c[ni] is None else sum_c[ni] + ec
                    sum_o[ni] = eo if sum_o[ni] is None else sum_o[ni] + eo
            for ni in noise_indices:
                mc = float((sum_c[ni] / args.n_noise_samples).mean())
                mo = float((sum_o[ni] / args.n_noise_samples).mean())
                scatter[ni][0].append(mc)
                scatter[ni][1].append(mo / mc if mc > 0 else float("nan"))
            n_collected += 1
        if n_collected >= args.n_scatter_samples:
            break

    # --- Finalize and save ---
    for k, ni in enumerate(noise_indices):
        fill_scatter_ax(axes[k, -1], scatter[ni][0], scatter[ni][1], labels[k])

    plt.tight_layout()
    save_path = os.path.join(output_dir, "epsilon_pixelwise.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
