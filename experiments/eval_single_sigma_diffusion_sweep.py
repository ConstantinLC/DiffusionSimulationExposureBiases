#!/usr/bin/env python
"""
Evaluate the two-stage pipeline: standalone Unet → single-step DiffusionModel.

For each sigma the pipeline is:
  1. Unet:          conditioning → x0_unet  (deterministic predictor, no noise)
  2. DiffusionModel: x0_unet re-noised at sigma_min → refined x0  (one DDPM step)

This is compared against the standalone Unet baseline (no DiffusionModel step).

Expected checkpoint layout:
  <unet_dir>/config.json + best_model.pth       (standalone Unet)
  <checkpoint_dir>/
    sigma_-4.0/DiffusionModel_single_-4.0_1/config.json + best_model.pth
    sigma_-3.9/DiffusionModel_single_-3.9_1/config.json + best_model.pth
    ...

Usage:
  python experiments/eval_single_sigma_diffusion_sweep.py \
      --unet_dir ./checkpoints/KuramotoSivashinsky/unet/Unet1D \
      --checkpoint_dir ./checkpoints/KuramotoSivashinsky/single_sigma_diffusion \
      --output results/single_sigma_diffusion_sweep.pdf
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel
from src.models.unet_1d import Unet1D
from src.models.unet_2d import Unet

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
}


# ── Model factories ────────────────────────────────────────────────────────────

def build_unet_model(model_cfg: dict) -> torch.nn.Module:
    """Instantiate a standalone Unet from its config dict."""
    model_type = model_cfg.get('type', 'Unet1D')
    if model_type == 'Unet1D':
        return Unet1D(
            dim=model_cfg.get('dim', model_cfg['dataSize'][0]),
            sigmas=torch.tensor(1),
            channels=model_cfg['condChannels'],
            dim_mults=tuple(model_cfg.get('dim_mults', [1, 1, 1])),
            use_convnext=True,
            convnext_mult=model_cfg.get('convnext_mult', 1),
            padding_mode=model_cfg.get('padding_mode', 'circular'),
            with_time_emb=model_cfg.get('with_time_emb', False),
        )
    elif model_type == 'Unet2D':
        return Unet(
            dim=model_cfg.get('dim', model_cfg['dataSize'][0]),
            sigmas=torch.zeros(1),
            channels=model_cfg['condChannels'],
            dim_mults=tuple(model_cfg.get('dim_mults', [1, 1, 1])),
            use_convnext=True,
            convnext_mult=model_cfg.get('convnext_mult', 1),
            padding_mode=model_cfg.get('padding_mode', 'circular'),
            with_time_emb=model_cfg.get('with_time_emb', False),
        )
    else:
        raise ValueError(f"Unknown Unet type: {model_type}")


def build_diffusion_model(model_cfg: dict) -> DiffusionModel:
    """Instantiate a DiffusionModel from its config dict."""
    return DiffusionModel(
        dimension=model_cfg['dimension'],
        dataSize=model_cfg['dataSize'],
        condChannels=model_cfg['condChannels'],
        dataChannels=model_cfg['dataChannels'],
        diffSchedule=model_cfg['diffSchedule'],
        diffSteps=model_cfg['diffSteps'],
        inferenceSamplingMode=model_cfg['inferenceSamplingMode'],
        inferenceConditioningIntegration=model_cfg['inferenceConditioningIntegration'],
        diffCondIntegration=model_cfg['diffCondIntegration'],
        padding_mode=model_cfg.get('padding_mode', 'circular'),
        architecture=model_cfg.get('architecture', 'ours'),
    )


# ── Directory helpers ──────────────────────────────────────────────────────────

def find_diffusion_run_dir(sigma_dir: str) -> str | None:
    """Return the DiffusionModel run directory inside sigma_dir, if completed."""
    try:
        runs = [d for d in os.listdir(sigma_dir) if d.startswith('DiffusionModel_')]
    except FileNotFoundError:
        return None
    runs_with_ckpt = [
        r for r in runs
        if os.path.exists(os.path.join(sigma_dir, r, 'best_model.pth'))
    ]
    if not runs_with_ckpt:
        return None
    return os.path.join(sigma_dir, runs_with_ckpt[-1])


def collect_sigma_dirs(checkpoint_dir: str) -> list[tuple[float, str]]:
    """Return sorted (log_sigma, run_dir) pairs for all finished DiffusionModel runs."""
    entries = []
    for name in os.listdir(checkpoint_dir):
        if not name.startswith('sigma_'):
            continue
        log_sigma = float(name[len('sigma_'):])
        sigma_dir = os.path.join(checkpoint_dir, name)
        run_dir = find_diffusion_run_dir(sigma_dir)
        if run_dir is None:
            print(f"  [skip] {name}: no completed DiffusionModel run found")
            continue
        entries.append((log_sigma, run_dir))
    return sorted(entries, key=lambda x: x[0])


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_unet(unet_model, val_loader, device) -> float:
    """MSE of the standalone Unet output vs ground truth, averaged over val set."""
    unet_model.eval()
    mse_list = []
    with torch.no_grad():
        for sample in val_loader:
            data = sample["data"].to(device)
            cond, target = data[:, 0], data[:, 1]
            x0_unet = unet_model(cond, time=None)
            mse_list.append(torch.mean((x0_unet - target) ** 2).item())
    return float(np.mean(mse_list))


def evaluate_diffusion_refinement(unet_model, diff_model, val_loader, device) -> float:
    """
    MSE of the two-stage output vs ground truth, averaged over val set.

    Stage 1: x0_unet = Unet(cond)
    Stage 2: re-noise x0_unet at sigma_min, then run one DiffusionModel step.
    """
    unet_model.eval()
    diff_model.eval()
    mse_list = []
    with torch.no_grad():
        for sample in val_loader:
            data = sample["data"].to(device)
            cond, target = data[:, 0], data[:, 1]
            B = cond.shape[0]

            # Stage 1: Unet prediction (deterministic)
            x0_unet = unet_model(cond, time=None)

            # Stage 2: single denoising step from x0_unet
            t = torch.zeros(B, device=device, dtype=torch.long)
            alpha = diff_model.sqrtAlphasCumprod[t]           # (B, 1, 1) or (B, 1, 1, 1)
            sigma = diff_model.sqrtOneMinusAlphasCumprod[t]   # same shape

            # Re-noise the Unet output at the DiffusionModel's sigma level
            noisy = alpha * x0_unet + sigma * torch.randn_like(x0_unet)

            # One denoising step
            inp = torch.cat((cond, noisy), dim=1)
            pred_noise = diff_model.unet(inp, t)[:, cond.shape[1]:]
            x0_two_stage = (noisy - sigma * pred_noise) / alpha

            mse_list.append(torch.mean((x0_two_stage - target) ** 2).item())
    return float(np.mean(mse_list))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet_dir', required=True,
                        help='Run dir of the standalone Unet (contains config.json + best_model.pth)')
    parser.add_argument('--checkpoint_dir',
                        default='./checkpoints/KuramotoSivashinsky/single_sigma_diffusion',
                        help='Root dir of the single-step DiffusionModel sigma sweep')
    parser.add_argument('--output', default='results/single_sigma_diffusion_sweep.pdf')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Load standalone Unet ───────────────────────────────────────────────────
    unet_config_path = os.path.join(args.unet_dir, 'config.json')
    with open(unet_config_path) as f:
        unet_config = json.load(f)

    unet_model = build_unet_model(unet_config['model_params']).to(args.device)
    unet_ckpt = os.path.join(args.unet_dir, 'best_model.pth')
    unet_model.load_state_dict(torch.load(unet_ckpt, map_location=args.device))
    print(f"Loaded Unet from {unet_ckpt}")

    # ── Discover DiffusionModel runs ───────────────────────────────────────────
    entries = collect_sigma_dirs(args.checkpoint_dir)
    if not entries:
        print("No completed DiffusionModel runs found. Exiting.")
        return
    print(f"Found {len(entries)} sigma runs: {[e[0] for e in entries]}")

    # ── Build val_loader from the first DiffusionModel config ─────────────────
    first_config_path = os.path.join(entries[0][1], 'config.json')
    with open(first_config_path) as f:
        first_config = json.load(f)

    raw_data = first_config['data_params']
    data_cfg = DataConfig(**{
        **{k: v for k, v in raw_data.items() if k in _DATA_CONFIG_FIELDS},
        'batch_size':     args.batch_size,
        'val_batch_size': args.batch_size,
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    # ── Compute Unet baseline MSE once ─────────────────────────────────────────
    print("\nEvaluating standalone Unet baseline ...")
    mse_unet = evaluate_unet(unet_model, val_loader, args.device)
    print(f"  MSE Unet: {mse_unet:.4e}")

    # ── Evaluate each sigma ────────────────────────────────────────────────────
    log_sigmas = []
    mse_two_stage_vals = []

    for log_sigma, run_dir in entries:
        print(f"\n── sigma = 10^{log_sigma:.1f}  ({run_dir}) ──")

        with open(os.path.join(run_dir, 'config.json')) as f:
            config = json.load(f)

        diff_model = build_diffusion_model(config['model_params']).to(args.device)
        ckpt_path = os.path.join(run_dir, 'best_model.pth')
        diff_model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        mse_ts = evaluate_diffusion_refinement(unet_model, diff_model, val_loader, args.device)
        print(f"  MSE two-stage: {mse_ts:.4e}  (vs Unet: {mse_unet:.4e})")

        log_sigmas.append(log_sigma)
        mse_two_stage_vals.append(mse_ts)

    # ── Plot ───────────────────────────────────────────────────────────────────
    log_sigmas_arr = np.array(log_sigmas)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.axhline(mse_unet, color='gray', linestyle='--', linewidth=1.5,
               label=f'Unet alone  ({mse_unet:.2e})')
    ax.plot(log_sigmas_arr, mse_two_stage_vals, 'o-', color='blue',
            label='Unet + DiffusionModel (two-stage)')

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(\sigma)$', fontsize=12)
    ax.set_ylabel('MSE vs ground truth', fontsize=12)
    ax.set_title('Two-stage pipeline: Unet + single-step DiffusionModel')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"\nPlot saved to {args.output}")
    plt.close(fig)


if __name__ == '__main__':
    main()
