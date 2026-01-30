#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from torch import nn

import sys
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

# --- Project Imports ---
from src.data_loader import get_data_loaders
from src.dataset import TurbulenceDataset
from src.data_transformations import DataParams, Transforms
from src.model_diffusion import DiffusionModel
from src.utils import count_parameters, parse_checkpoint_args, run_model
from torch.utils.data import DataLoader

# ==========================================
# 1. Style Configuration (Times Font)
# ==========================================
sns.set_theme(style="white", context="paper", font_scale=1.4)
rcParams = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman"],
    "mathtext.fontset": "stix",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times} \usepackage{amsmath} \usepackage{amssymb}",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
}
mpl.rcParams.update(rcParams)


def evaluate_dw_train_inf_gap(models, val_loader, device):
    """
    Runs autoregressive rollout for one model over the FULL dataset.
    """
    for model in list(models.values()):
        model.eval()
    
    total_samples = 0
    print(f"Starting evaluation over full dataset ({len(val_loader)} batches)...")

    with torch.no_grad():
        mse_clean_all = {name: [] for name in models}
        mse_clean_own_pred_all = {name: [] for name in models}
        mse_clean_prev_pred_all = {name: [] for name in models}

        for batch_idx, sample in enumerate(val_loader):
            data = sample["data"].to(device)
            batch_size = data.shape[0]
            total_samples += batch_size

            conditioning_frame = data[:,0]
            target_frame = data[:, 1]

            for name in models:
                model = models[name]
                
                # 1. Clean (Teacher Forcing)
                _, x0_estimates_clean = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="clean")
                
                # 2. Own Pred (Self-Conditioning)
                _, x0_estimates_clean_own_pred = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="own-pred")
                
                # 3. Prev Pred (Two-Step Lookahead)
                _, x0_estimates_clean_prev_pred = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="prev-pred")

                # Compute MSEs
                mse_clean = [(torch.mean((x0_estimates_clean[t] - target_frame)**2)).item() for t in range(len(x0_estimates_clean))]
                mse_clean_own_pred = [(torch.mean((x0_estimates_clean_own_pred[t] - target_frame)**2)).item() for t in range(len(x0_estimates_clean))]
                mse_clean_prev_pred = [(torch.mean((x0_estimates_clean_prev_pred[t] - target_frame)**2)).item() for t in range(len(x0_estimates_clean))]
                
                mse_clean_all[name].append(mse_clean)
                mse_clean_own_pred_all[name].append(mse_clean_own_pred)
                mse_clean_prev_pred_all[name].append(mse_clean_prev_pred)

            if batch_idx == 0:
                break

    mean_mse_clean = {}
    mean_mse_clean_own_pred = {}
    mean_mse_clean_prev_pred = {}

    for name in models:
        mean_mse_clean[name] = torch.mean(torch.tensor(mse_clean_all[name]), dim=0)
        mean_mse_clean_own_pred[name] = torch.mean(torch.tensor(mse_clean_own_pred_all[name]), dim=0)
        mean_mse_clean_prev_pred[name] = torch.mean(torch.tensor(mse_clean_prev_pred_all[name]), dim=0)

    return {
        "mse_clean": mean_mse_clean,
        "mse_clean_own_pred": mean_mse_clean_own_pred,
        "mse_clean_prev_pred": mean_mse_clean_prev_pred,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models on Turbulence Data")
    parser.add_argument('--config', type=str, required=True, help="Config file")
    parser.add_argument('--checkpoints', nargs='+', required=True, help="List of checkpoints.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, 'r') as f: config = json.load(f)

    # Parse Checkpoints
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)
    first_key = list(checkpoints_dict.keys())[0]
    checkpoints_dict = {first_key: checkpoints_dict[first_key]}

    print(f"--- Evaluating Single Model: {first_key} ---")

    config['data_params']['batch_size']=128
    config['data_params']['validation_batch_size']=128

    _, _, testLoader = get_data_loaders(config['data_params'])

    models = {}
    for name, ckpt_path in checkpoints_dict.items():
        print(f"--- Loading Candidate: {name} ---")
        model_config = config['model_params']
        model_config['checkpoint'] = ckpt_path
        model_config['load_betas'] = True
        model = DiffusionModel(**model_config).to(args.device)
        models[name] = model

    results = evaluate_dw_train_inf_gap(models, testLoader, device=args.device)

    # --- Data Processing for Seaborn ---
    model_name = first_key
    model = models[model_name]
    
    raw_alphas = model.sqrtOneMinusAlphasCumprod.ravel().cpu().numpy()
    
    # Reverse MSEs to match Low->High sigma order
    mse_clean = np.array(results['mse_clean'][model_name])[::-1]
    mse_own = np.array(results['mse_clean_own_pred'][model_name])[::-1]
    mse_prev = np.array(results['mse_clean_prev_pred'][model_name])[::-1]

    # Sort based on alphas
    sort_idx = np.argsort(raw_alphas)
    alphas_sorted = raw_alphas[sort_idx]
    mse_clean_sorted = mse_clean[sort_idx]
    mse_own_sorted = mse_own[sort_idx]
    mse_prev_sorted = mse_prev[sort_idx]

    # Filter for Low Noise Regime (first 6 points)
    slice_indices = range(19)
    
    # Construct DataFrame with RATIOS
    data_list = []
    for i in slice_indices:
        sigma = alphas_sorted[i]
        
        # Calculate Ratios
        ratio_own = mse_own_sorted[i] / mse_clean_sorted[i]
        ratio_prev = mse_prev_sorted[i] / mse_clean_sorted[i]
        
        data_list.append({"Noise Level": sigma, "Ratio": ratio_own, "Method": "Ratio: Own-Pred / Clean"})
        data_list.append({"Noise Level": sigma, "Ratio": ratio_prev, "Method": "Ratio: Two-Steps / Clean"})

    df = pd.DataFrame(data_list)

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    
    markers = {
        "Ratio: Own-Pred / Clean": "^", 
        "Ratio: Two-Steps / Clean": "X"
    }
    
    linestyles = {
        "Ratio: Own-Pred / Clean": (3, 1, 1, 1), # Dashdot
        "Ratio: Two-Steps / Clean": (2, 2)       # Dashed
    }
    
    ax = sns.lineplot(
        data=df, 
        x="Noise Level", 
        y="Ratio", 
        hue="Method", 
        style="Method", 
        markers=markers,
        dashes=linestyles, 
        linewidth=2.5,
        markersize=9,
        palette="deep"
    )

    # --- Apply Styling ---
    ax.set_xlabel(r"Noise Level $\mathbf{\sigma}$", labelpad=10, fontsize=17)
    # y-axis label for Ratio
    ax.set_ylabel(r"Error Ratio ($\mathcal{E} / \mathcal{E}_{\text{clean}}$)", labelpad=10, fontsize=17)
    
    # Scales
    ax.set_xscale('log')
    # Using linear scale often better for ratios close to 1, but log if they diverge
    # If the gap is small, linear is better. If gap is huge, log. 
    # Let's keep linear for ratios unless requested otherwise, as they are relative.
    # But usually gap plots are log-log. Let's stick to log-log to be safe.
    # ax.set_yscale('log') 
    
    # Frame/Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
        
    # Grid
    ax.grid(True, which="major", ls="-", alpha=0.3, color='grey')
    ax.grid(True, which="minor", ls=":", alpha=0.2, color='grey')
    
    # Legend
    legend = ax.legend(
        title=r"\textbf{Metric}", 
        fontsize=11, 
        title_fontsize=12,
        loc='upper left',
        frameon=True,
        fancybox=False,
        edgecolor='black'
    )

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "train-inf-gap_ratios.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()