#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch import nn

import sys
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

# --- Project Imports ---
from src.data_loader import get_data_loaders
from src.dataset import TurbulenceDataset
from src.data_transformations import DataParams, Transforms
from src.model_diffusion import DiffusionModel
from src.model import Unet
from src.utils import count_parameters, parse_checkpoint_args, run_model
from torch.utils.data import DataLoader, SequentialSampler


def evaluate_dw_train_inf_gap(models, val_loader, device):
    """
    Runs autoregressive rollout for one model over the FULL dataset.
    """
    # 1. Set models to eval mode
    for model in list(models.values()):
        model.eval()
    
    total_samples = 0
    print(f"Starting evaluation over full dataset ({len(val_loader)} batches)...")

    with torch.no_grad():
        mse_ancestor_all = {name: [] for name in models}
        mse_clean_all = {name: [] for name in models}
        mse_clean_own_pred_all = {name: [] for name in models}

        for batch_idx, sample in enumerate(val_loader):
            # --- A. Prepare Batch ---
            data = sample["data"].to(device) # (B, T_total, C, H, W)
            batch_size = data.shape[0]
            total_samples += batch_size

            # Initial Condition (t=0)
            conditioning_frame = data[:,0]
            target_frame = data[:, 1]

            for name in models:
                # Store prediction
                model = models[name]
                _, x0_estimates = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="ancestor")
                _, x0_estimates_clean = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="clean")
                _, x0_estimates_clean_own_pred = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="own-pred")

                mse_ancestor = [(torch.mean((x0_estimates[t] - target_frame)**2)).item()
                        for t in range(len(x0_estimates))]
                mse_clean = [(torch.mean((x0_estimates_clean[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]
                mse_clean_own_pred = [(torch.mean((x0_estimates_clean_own_pred[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]
                
                mse_ancestor_all[name].append(mse_ancestor)
                mse_clean_all[name].append(mse_clean)
                mse_clean_own_pred_all[name].append(mse_clean_own_pred)

            if batch_idx == 0:
                break

    # 3. Aggregate results
    mean_mse_ancestor = {}
    mean_mse_clean = {}
    mean_mse_clean_own_pred = {}

    for name in models:
        # Concatenate all batches along dimension 0
        mean_mse_ancestor[name] = torch.mean(torch.tensor(mse_ancestor_all[name]), dim=0)
        mean_mse_clean[name] = torch.mean(torch.tensor(mse_clean_all[name]), dim=0)
        mean_mse_clean_own_pred[name] = torch.mean(torch.tensor(mse_clean_own_pred_all[name]), dim=0)

    return {
        "mse_ancestor": mean_mse_ancestor,
        "mse_clean": mean_mse_clean,
        "mse_clean_own_pred": mean_mse_clean_own_pred,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models on Turbulence Data")
    
    # Paths
    parser.add_argument('--config', type=str, required=True, help="Config file")
    
    # Checkpoints
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                        help="List of checkpoints. Format: 'Name=/path/to/ckpt.pth'. Space separated.")
    
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots and metrics")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, 'r') as f: config = json.load(f)

    # 0. Parse Checkpoints into Dictionary
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)
    
    # FORCE SINGLE MODEL: Take only the first one if multiple are provided
    first_key = list(checkpoints_dict.keys())[0]
    checkpoints_dict = {first_key: checkpoints_dict[first_key]}

    print(f"--- Evaluating Single Model: {first_key} ---")

    config['data_params']['batch_size']=16
    config['data_params']['validation_batch_size']=16

    _, _, testLoader = get_data_loaders(config['data_params'])

    # 2. Load Candidate Model
    models = {}
    for name, ckpt_path in checkpoints_dict.items():
        print(f"--- Loading Candidate: {name} ---")
        model_config = config['model_params']
        model_config['checkpoint'] = ckpt_path
        model_config['load_betas'] = True
        # Initialize Architecture
        model = DiffusionModel(
            **model_config
        ).to(args.device)
        
        models[name] = model

    # 3. Evaluate train input vs inference input predictions
    results = evaluate_dw_train_inf_gap(models, testLoader, device=args.device)

    # 4. Plotting for Single Model
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    
    color_main = 'purple'
    model_name = first_key
    model = models[model_name]
    
    mse_ancestor = results['mse_ancestor'][model_name]
    mse_clean = results['mse_clean'][model_name]
    mse_clean_own_pred = results['mse_clean_own_pred'][model_name]
    
    # Get alphas/sigmas
    alphas = list(model.sqrtOneMinusAlphasCumprod.ravel().cpu())[::-1]

    # Subplot 0: Histogram
    axes[0].hist(alphas, alpha=0.3, color=color_main, bins=np.logspace(-2.5, 0, 20))
    axes[0].set_title(f"Noise Distribution: {model_name}")
    axes[0].set_ylabel('Count')
    axes[0].set_xscale('log')

    # Subplot 1: MSE Curves
    axes[1].plot(alphas, mse_clean, label="Training input (Clean)", 
                 color='blue', linestyle='dotted', linewidth=2)
    axes[1].plot(alphas, mse_ancestor, label="Inference input (Ancestor)", 
                 color='red', linewidth=2)
    axes[1].plot(alphas, mse_clean_own_pred, label="Inference input (Own Pred)", 
                 color='green', linestyle='dashdot', linewidth=2)

    axes[1].grid(True, which='both', linestyle='--', alpha=0.3)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('MSE w/ Ground Truth')
    axes[1].set_xlabel('Noise Level $\sqrt{1-\\bar{\\alpha}_t}$ (Log Scale)')
    axes[1].legend(fontsize=10)

    # Add text summary
    summary_text = (
        f"Final Errors:\n"
        f"Training (Clean):  {mse_clean[-1]:.2e}\n"
        f"Inference (Ancestor): {mse_ancestor[-1]:.2e}"
    )
    
    # Place text box in the plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes, 
                 fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "train-inf-gap_single.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()