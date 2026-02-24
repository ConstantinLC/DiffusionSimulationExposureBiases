#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch import nn
import matplotlib.lines as mlines
from matplotlib.ticker import LogFormatter

import sys
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

# --- Project Imports ---
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.model import Unet
from src.utils import count_parameters, parse_checkpoint_args, run_model
from src.diffusion_utils import evaluate_dw_train_inf_gap

colors = ['purple', 'blue', 'red', 'green']

# --- CUSTOM SCALE FUNCTION ---
def get_custom_scale(axis_min=-6, axis_break=-3, axis_max=-1, break_pos=0.75):
    def forward(x):
        x_safe = np.array(x, dtype=float)
        x_safe[x_safe <= 0] = 10**(axis_min - 1) 
        lx = np.log10(x_safe)
        norm_low = (lx - axis_min) / (axis_break - axis_min)
        out_low = norm_low * break_pos
        norm_high = (lx - axis_break) / (axis_max - axis_break)
        out_high = break_pos + norm_high * (1 - break_pos)
        return np.where(lx < axis_break, out_low, out_high)

    def inverse(y):
        lx_low = (y / break_pos) * (axis_break - axis_min) + axis_min
        lx_high = ((y - break_pos) / (1 - break_pos)) * (axis_max - axis_break) + axis_break
        lx = np.where(y < break_pos, lx_low, lx_high)
        return 10**lx
    return forward, inverse


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models on Turbulence Data")
    
    # UPDATED ARGUMENT:
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                        help="List of checkpoints. Format: 'Name=/path/to/ckpt.pth'. Space separated.")
    
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots and metrics")
    
    parser.add_argument('--limit_val_trajectories', type=int, default=1) 
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 0. Parse Checkpoints into Dictionary
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)
    print(f"--- Evaluating {len(checkpoints_dict)} Models ---")
    for name, path in checkpoints_dict.items():
        print(f"  > {name}: {path}")

    # 1. Load Data
    print("--- Loading Data ---")
    data_params = {
        "base_checkpoint_dir": "./checkpoints",
        "dataset_name": "KuramotoSivashinsky",
        "data_path": "/mnt/SSD2/constantin/archives/LPSDA/data_og",
        "resolution": 64,
        "sequence_length": [2, 4],
        "trajectory_sequence_length": [160, 4],
        "limit_trajectories_train": 1000,
        "limit_trajectories_val": 20,
        "batch_size": 64,
        "val_batch_size": 64
    }
    _, val_loader, _ = get_data_loaders(data_params)

    # 2. Load Candidate Models (Looping over Dictionary)
    models = {}
    for name, ckpt_path in checkpoints_dict.items():
        print(f"--- Loading Candidate: {name} ---")
        
        # Initialize Architecture
        model = DiffusionModel(
            dimension=1,
            dataSize=[64, 64],
            condChannels=1,
            dataChannels=1,
            diffSchedule="linear",
            diffSteps=20,
            inferenceSamplingMode="ddpm",
            inferenceConditioningIntegration="clean",
            diffCondIntegration="clean",
            inferenceInitialSampling="random",
            architecture="Unet1D",
            checkpoint=ckpt_path,
            load_betas=True
        ).to(args.device)
            
        models[name] = model

    # 3. Evaluate train input vs inference input predictions
    results = evaluate_dw_train_inf_gap(models, val_loader, device='cuda')

    n_models = len(checkpoints_dict.items())
    fig, axes = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True, 
                         gridspec_kw={'height_ratios': [1, 1.5]})

    colors = ['purple', 'blue', 'red']

    # --------------------
    for i, model_name in enumerate(models):

        model = models[model_name]
        mse_ancestor = results['mse_ancestor'][model_name]
        mse_clean = results['mse_clean'][model_name]
        mse_own_pred = results['mse_clean_own_pred'][model_name]
        alphas = list(models[model_name].sqrtOneMinusAlphasCumprod.ravel().cpu())[::-1]

        color = colors[i % len(colors)]

        # --- TOP PLOT ---
        bins = np.logspace(-1.85, 0, 20)
        labels = [ "LowLevels-Focus", "Linear", "Cubic"]
        axes[0].hist(alphas, alpha=0.2, color=color, bins=bins)
        axes[0].hist(alphas, histtype='step', color=color, bins=bins, linewidth=1.5, label=labels[i])

        # --- MIDDLE PLOT ---
        axes[1].plot(alphas, mse_clean, color=color, linestyle='dotted', linewidth=2)
        axes[1].plot(alphas, mse_ancestor, color=color, linestyle='solid', linewidth=2)
        axes[1].plot(alphas, mse_own_pred, color=color, linestyle='dashdot', linewidth=2)

        # --- TEXT LABELS ---
        axes[1].text(
            0.07 + i * 0.3,  0.09,
            r"$\mathcal{E}_{\text{clean}}(0)$" + f"= {mse_clean[-1]:.1e}",
            transform=axes[1].transAxes,
            fontsize=8, color=color, fontweight='bold', verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )
        axes[1].text(
            0.07 + i * 0.3,  0.05,
            r"$\mathcal{E}_{\text{inf}}(0)$" + f"= {mse_ancestor[-1]:.1e}",
            transform=axes[1].transAxes,
            fontsize=8, color=color, fontweight='bold', verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )
        axes[1].text(
        0.07 + i * 0.3,  0.01,
        r"$\mathcal{E}_{\text{own}}(0)$" + f"= {mse_own_pred[-1]:.1e}",
        transform=axes[1].transAxes,
        fontsize=8, color=color, fontweight='bold', verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )


    # --------------------
    # VISUALS & ARROW
    # --------------------

    # --- 1. ADD SAMPLING ARROW ---
    # We use 'axes fraction' coords: (0,0) is bottom-left, (1,1) is top-right of the subplot
    # Arrow from Right (0.95) to Left (0.05)
    axes[0].annotate(
        '', xy=(0.02, 1.15), xytext=(0.98, 1.15),           # xy=Tip (Left), xytext=Tail (Right)
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", color='black', lw=1.5)
    )
    # Text label centered above the arrow
    axes[0].text(
        0.5, 1.2, r"Sampling direction, $T=100$", 
        transform=axes[0].transAxes, ha='center', va='bottom', fontsize=10
    )

    # --- Top Plot formatting ---
    axes[0].set_xscale('log')
    axes[0].set_ylabel("Schedule Distribution", fontsize=13)
    axes[0].tick_params(labelbottom=False)

    # --- Middle Plot formatting ---
    forward, inverse = get_custom_scale(axis_min=-8, axis_break=-5, axis_max=-1, break_pos=0.75)
    axes[1].set_yscale('function', functions=(forward, inverse))
    axes[1].set_yticks([1e-7, 1e-5, 1e-3])
    axes[1].yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    axes[1].set_ylim(1e-8, 1e-2) 
    axes[1].grid(True, which="major", alpha=0.3)
    axes[1].set_ylabel(r"MSE $\mathcal{E}$ with target", fontsize=13)

    # Legend
    line_clean = mlines.Line2D([], [], color='black', linestyle='dotted', linewidth=2, label='Clean Input')
    line_ancestor = mlines.Line2D([], [], color='black', linestyle='solid', linewidth=2, label='Inference Input')
    line_own = mlines.Line2D([], [], color='black', linestyle='dashdot', linewidth=2, label='Own-Prediction Input')

    axes[0].legend(loc='upper left', fontsize=9, frameon=True)
    axes[1].legend(handles=[line_clean, line_ancestor, line_own], loc='upper left', fontsize=9, frameon=True)

    axes[0].set_yscale('log')

    # --- Layout ---
    fig.align_ylabels(axes)
    plt.subplots_adjust(hspace=0.1, top=0.9) # Increase top margin slightly for the arrow
    plt.savefig(os.path.join(args.output_dir, "train-inf-gap.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
