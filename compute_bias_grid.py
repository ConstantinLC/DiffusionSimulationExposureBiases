"""
Compute the two-step bias grid B^(2S)(sigma_x, sigma_y) for all sigma_x < sigma_y,
using a single checkpoint.

The result is an (n_noise_levels x n_noise_levels) matrix with entries only above
the diagonal (position [i,j] with i<j means sigma_x=sigmas[i], sigma_y=sigmas[j]).

Usage:
    python compute_bias_grid.py [--checkpoint PATH] [--n_batches N] [--n_noise_samples N]
                                [--output_dir PATH] [--device cuda|cpu]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.diffusion import DiffusionModel
from src.data.loaders import get_data_loaders
from src.config import DataConfig

CHECKPOINT_PATH = (
    "/mnt/SSD2/constantin/diffusion-multisteps/checkpoints/"
    "KolmogorovFlow/DiffusionModel_cosine/run_0/best_model.pth"
)
CONFIG_PATH = (
    "/mnt/SSD2/constantin/diffusion-multisteps/checkpoints/"
    "KolmogorovFlow/DiffusionModel_cosine/run_0/config.json"
)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(cfg):
    mp = cfg["model_params"]
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
        padding_mode=mp["padding_mode"],
        architecture=mp["architecture"],
        checkpoint="",  # weights loaded separately
        load_betas=False,
    )


# ---------------------------------------------------------------------------
# Two-step bias (single checkpoint)
# ---------------------------------------------------------------------------

def compute_two_step_bias(
    model, checkpoint_state,
    sigma_high, sigma_low,
    val_loader, device, n_batches=30, n_noise_samples=1,
):
    """
    Compute B^(2S)(sigma_low, sigma_high) using a single checkpoint for both steps.

    First denoises from sigma_high -> x0_hat, then re-noises to sigma_low and
    denoises again. The ratio of two-step MSE to clean MSE at sigma_low is returned.
    """
    model.eval()
    schedule = torch.tensor([sigma_low, sigma_high], dtype=torch.float32)
    IDX_LOW, IDX_HIGH = 0, 1

    _SCHEDULE_KEYS = {"sqrtAlphasCumprod", "sqrtOneMinusAlphasCumprod", "unet.sigmas"}

    # Load weights and set schedule (single checkpoint, done once)
    model.load_state_dict(
        {k: v for k, v in checkpoint_state.items() if k not in _SCHEDULE_KEYS},
        strict=False,
    )
    model.compute_schedule_variables(schedule)
    model = model.to(device)

    sigmas = model.sqrtOneMinusAlphasCumprod
    sqrt_alpha = model.sqrtAlphasCumprod

    # --- Phase A: denoise all batches at sigma_high -> x0_hat ---
    x0_hats, targets_list, conds_list = [], [], []
    spatial_dims = None

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            if batch_idx >= n_batches:
                break
            data = sample["data"].to(device)
            cond = data[:, 0]
            target = data[:, 1]
            if spatial_dims is None:
                spatial_dims = tuple(range(1, target.ndim))
            N = target.shape[0]
            t_high = torch.full((N,), IDX_HIGH, device=device, dtype=torch.long)

            x0_hat_acc = torch.zeros_like(target)
            for _ in range(n_noise_samples):
                eps = torch.randn_like(target)
                y_noisy = sqrt_alpha[t_high] * target + sigmas[t_high] * eps
                pred = model.unet(torch.cat((cond, y_noisy), dim=1), t_high)[:, cond.shape[1]:]
                x0_hat_acc += (y_noisy - sigmas[t_high] * pred) / sqrt_alpha[t_high]
            x0_hat_acc /= n_noise_samples

            x0_hats.append(x0_hat_acc.cpu())
            targets_list.append(target.cpu())
            conds_list.append(cond.cpu())

    # --- Phase B: re-noise x0_hat to sigma_low, denoise, compare ---
    all_twostep_mse, all_clean_mse = [], []

    with torch.no_grad():
        for x0_hat, target, cond in zip(x0_hats, targets_list, conds_list):
            x0_hat = x0_hat.to(device)
            target = target.to(device)
            cond = cond.to(device)
            N = target.shape[0]
            t_low = torch.full((N,), IDX_LOW, device=device, dtype=torch.long)

            twostep_acc = torch.zeros(N, device=device)
            clean_acc = torch.zeros(N, device=device)
            for _ in range(n_noise_samples):
                # Two-step path: denoise x0_hat
                eps_low = torch.randn_like(target)
                y_noisy_low = sqrt_alpha[t_low] * x0_hat + sigmas[t_low] * eps_low
                pred_low = model.unet(torch.cat((cond, y_noisy_low), dim=1), t_low)[:, cond.shape[1]:]
                x0_final = (y_noisy_low - sigmas[t_low] * pred_low) / sqrt_alpha[t_low]
                twostep_acc += (x0_final - target).pow(2).mean(dim=spatial_dims)

                # Clean path: denoise from true target
                eps_clean = torch.randn_like(target)
                y_clean_low = sqrt_alpha[t_low] * target + sigmas[t_low] * eps_clean
                pred_clean = model.unet(torch.cat((cond, y_clean_low), dim=1), t_low)[:, cond.shape[1]:]
                x0_clean = (y_clean_low - sigmas[t_low] * pred_clean) / sqrt_alpha[t_low]
                clean_acc += (x0_clean - target).pow(2).mean(dim=spatial_dims)

            all_twostep_mse.append((twostep_acc / n_noise_samples).cpu())
            all_clean_mse.append((clean_acc / n_noise_samples).cpu())

    mean_twostep = torch.cat(all_twostep_mse).mean().item()
    mean_clean = torch.cat(all_clean_mse).mean().item()
    bias = mean_twostep / max(mean_clean, 1e-12)
    return bias, mean_clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--n_batches", type=int, default=30)
    parser.add_argument("--n_noise_samples", type=int, default=1)
    parser.add_argument("--output_dir", default=None,
                        help="defaults to the checkpoint's parent directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    # --- Config & model ---
    with open(args.config) as f:
        cfg = json.load(f)

    model = build_model(cfg)
    model = model.to(device)

    # Extract sigmas from the model's cosine schedule (sorted low -> high).
    # Drop the last level (sigma ≈ 1) — it is numerically degenerate.
    sigmas = model.sqrtOneMinusAlphasCumprod.ravel().cpu().numpy().copy()[:-1]
    n = len(sigmas)
    print(f"Schedule has {n} noise levels (last dropped)")
    print(f"  sigma range: [{sigmas[0]:.6f}, {sigmas[-1]:.6f}]")

    # --- Checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_state = torch.load(args.checkpoint, map_location=device)

    # --- Data ---
    dp = cfg["data_params"]
    data_cfg = DataConfig(
        dataset_name="KolmogorovFlow",
        data_path=dp["data_path"],
        resolution=dp["resolution"],
        frames_per_time_step=dp["frames_per_time_step"],
        limit_trajectories_train=dp["limit_trajectories_train"],
        limit_trajectories_val=dp["limit_trajectories_val"],
        batch_size=dp["batch_size"],
        val_batch_size=dp["val_batch_size"],
    )
    _, val_loader, _ = get_data_loaders(data_cfg)

    # --- Grid computation ---
    # bias_grid[i, j] = B^(2S)(sigmas[i], sigmas[j])  for i < j (sigma_x < sigma_y)
    bias_grid = np.full((n, n), np.nan)
    n_pairs = n * (n - 1) // 2
    done = 0

    for i in range(n):
        for j in range(i + 1, n):
            sigma_low = float(sigmas[i])
            sigma_high = float(sigmas[j])
            bias, clean_err = compute_two_step_bias(
                model, checkpoint_state,
                sigma_high, sigma_low,
                val_loader, device,
                n_batches=args.n_batches,
                n_noise_samples=args.n_noise_samples,
            )
            bias_grid[i, j] = bias
            done += 1
            print(
                f"[{done}/{n_pairs}] B^(2S)(sigma_x={sigma_low:.5f}, sigma_y={sigma_high:.5f})"
                f" = {bias:.4f}  (clean_err={clean_err:.3e})"
            )

    # --- Save ---
    np.save(os.path.join(output_dir, "bias_grid.npy"), bias_grid)
    np.save(os.path.join(output_dir, "bias_grid_sigmas.npy"), sigmas)
    print(f"\nSaved bias_grid.npy and bias_grid_sigmas.npy to {output_dir}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 7))
    log_grid = np.log10(bias_grid)
    masked = np.ma.masked_where(np.isnan(log_grid), log_grid)
    im = ax.imshow(masked, origin="lower", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label=r"$\log_{10}\,B^{(2S)}(\sigma_x, \sigma_y)$")

    tick_labels = [f"{s:.3f}" for s in sigmas]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_xlabel(r"$j$ (index of $\sigma_y$, high noise)")
    ax.set_ylabel(r"$i$ (index of $\sigma_x$, low noise)")
    ax.set_title(r"Two-step bias $B^{(2S)}(\sigma_x, \sigma_y)$, $\sigma_x < \sigma_y$ (log scale)")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bias_grid.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
