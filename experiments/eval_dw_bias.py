#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

from src.data.loaders import get_data_loaders
from src.config import DataConfig
from src.models.diffusion import DiffusionModel
from src.models.pderefiner import PDERefiner


# ─────────────────────────── model factory ───────────────────────────────────

def build_model(model_cfg: dict) -> torch.nn.Module:
    cls = model_cfg.get('type', None)
    if cls is None:
        if 'refinementSteps' in model_cfg:
            cls = 'PDERefiner'
        else:
            cls = 'DiffusionModel'
    print(cls)
    if cls == 'DiffusionModel':
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

    elif cls == 'PDERefiner':
        return PDERefiner(
            dimension=model_cfg['dimension'],
            dataSize=model_cfg['dataSize'],
            condChannels=model_cfg['condChannels'],
            dataChannels=model_cfg['dataChannels'],
            refinementSteps=model_cfg.get('refinementSteps', 3),
            log_sigma_min=model_cfg.get('log_sigma_min', -1.5),
            padding_mode=model_cfg.get('padding_mode', 'circular'),
            architecture=model_cfg.get('architecture', 'Unet2D'),
            multi_unet=model_cfg.get('multi_unet', False),
        )

    else:
        raise ValueError(f"Unknown model class '{cls}'")


# ─────────────────────────── noise-level helper ──────────────────────────────

def get_noise_levels(model): 
    """Return (noise_levels_list, x_axis_label) in inference order (high → low)."""
    if isinstance(model, DiffusionModel):
        levels = model.sqrtOneMinusAlphasCumprod.ravel().cpu().tolist()[::-1]
        label  = r'Noise Level $\sqrt{1-\bar{\alpha}_t}$ (Log Scale)'
    elif isinstance(model, PDERefiner):
        levels = model.sigmas.ravel().cpu().tolist()[::-1]
        label  = r'Noise Level $\sigma_t$ (Log Scale)'
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    return levels, label


# ─────────────────────────── evaluation ──────────────────────────────────────

def evaluate_dw_bias(model, val_loader, device):
    """
    Run one batch of validation data through the model with three different
    input_type settings and return per-timestep MSE lists.
    """
    model.eval()

    with torch.no_grad():
        sample = next(iter(val_loader))
        
        data              = sample["data"].to(device)   # (B, T, C, H, W)
        print(data.shape)
        conditioning_frame = data[:, 0]
        target_frame       = data[:, 1]

        _, x0_ancestor  = model(conditioning=conditioning_frame, data=target_frame,
                                return_x0_estimate=True, input_type="ancestor")
        _, x0_clean     = model(conditioning=conditioning_frame, data=target_frame,
                                return_x0_estimate=True, input_type="clean")
        _, x0_own_pred  = model(conditioning=conditioning_frame, data=target_frame,
                                return_x0_estimate=True, input_type="own-pred")

    mse = lambda estimates: [
        torch.mean((estimates[t] - target_frame) ** 2).item()
        for t in range(len(estimates))
    ]

    return {
        "mse_ancestor":       mse(x0_ancestor),
        "mse_clean":          mse(x0_clean),
        "mse_clean_own_pred": mse(x0_own_pred),
    }


# ─────────────────────────── plotting ────────────────────────────────────────

def plot_results(results, model, run_name, output_dir):
    noise_levels, x_label = get_noise_levels(model)
    noise_arr = np.array(noise_levels)

    mse_ancestor  = results['mse_ancestor']
    mse_clean     = results['mse_clean']
    mse_own_pred  = results['mse_clean_own_pred']

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # --- top: histogram of noise levels ---
    bins = np.logspace(np.log10(noise_arr.min()), np.log10(noise_arr.max()), 20)
    axes[0].hist(noise_levels, bins=bins, alpha=0.4, color='purple')
    axes[0].set_title(f"Noise Distribution: {run_name}")
    axes[0].set_ylabel('Count')
    axes[0].set_xscale('log')

    # --- bottom: MSE vs noise level ---
    axes[1].plot(noise_levels, mse_clean,    label="Training input (Clean)",
                 color='blue',  linestyle='dotted',   linewidth=2)
    axes[1].plot(noise_levels, mse_ancestor, label="Inference input (Ancestor)",
                 color='red',   linestyle='solid',    linewidth=2)
    axes[1].plot(noise_levels, mse_own_pred, label="Inference input (Own Pred)",
                 color='green', linestyle='dashdot',  linewidth=2)

    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].grid(True, which='both', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('MSE w/ Ground Truth')
    axes[1].set_xlabel(x_label)
    axes[1].legend(fontsize=10)

    summary = (
        f"Final Errors:\n"
        f"  Clean:    {mse_clean[-1]:.2e}\n"
        f"  Ancestor: {mse_ancestor[-1]:.2e}\n"
        f"  Own-Pred: {mse_own_pred[-1]:.2e}"
    )
    axes[1].text(0.05, 0.95, summary, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "dw_bias.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close(fig)


# ─────────────────────────── main ────────────────────────────────────────────

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution',
    'batch_size', 'val_batch_size',
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate DW bias for a saved experiment run")
    parser.add_argument('--run_dir', type=str, required=True,
                        help="Experiment folder containing config.json and checkpoints")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for evaluation (default: 16)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Where to save the plot (default: run_dir)")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    config_path = os.path.join(args.run_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    print(f"Run: {run_name}  |  model class: {config['model_params'].get('class', 'DiffusionModel')}")

    # ── Data loader (val split only) ─────────────────────────────────────────
    raw_data = config['data_params']
    data_cfg = DataConfig(**{
        **{k: v for k, v in raw_data.items() if k in _DATA_CONFIG_FIELDS},
        'batch_size':     args.batch_size,
        'val_batch_size': args.batch_size,
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    # ── Build & load model ───────────────────────────────────────────────────
    model = build_model(config['model_params']).to(args.device)
    ckpt_path = os.path.join(args.run_dir, 'best_model.pth')
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("Evaluating...")
    results = evaluate_dw_bias(model, val_loader, args.device)

    print(f"  MSE ancestor:  {results['mse_ancestor'][-1]:.4e}")
    print(f"  MSE clean:     {results['mse_clean'][-1]:.4e}")
    print(f"  MSE own-pred:  {results['mse_clean_own_pred'][-1]:.4e}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_results(results, model, run_name, output_dir)


if __name__ == "__main__":
    main()
