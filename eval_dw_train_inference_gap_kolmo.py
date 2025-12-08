#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch import nn

# --- Project Imports ---
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.model import Unet
from src.utils import count_parameters, parse_checkpoint_args, run_model, run_model


def evaluate_dw_train_inf_gap(models, val_loader, device):
    """
    Runs autoregressive rollout for multiple models over the FULL dataset.
    """
    # 1. Set models to eval mode
    for model in list(models.values()):
        model.eval()
    
    total_samples = 0

    print(f"Starting evaluation over full dataset ({len(val_loader)} batches)...")

    with torch.no_grad():

        mse_ancestor_all = {name: [] for name in models}
        mse_clean_all = {name: [] for name in models}

        for batch_idx, sample in enumerate(val_loader):
            
            # --- A. Prepare Batch ---
            data = sample["data"].to(device) # (B, T_total, C, H, W)
            batch_size = data.shape[0]
            total_samples += batch_size

            # Initial Condition (t=0)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]

            for name in models:
                # Store prediction
                model = models[name]
                _, x0_estimates, _ = model(conditioning=conditioning_frame, data=target_frame)
                _, x0_estimates_clean, _ = model(conditioning=conditioning_frame, data=target_frame, with_clean_input=True)

                mse_ancestor = [(torch.mean((x0_estimates[t].cpu() - target_frame)**2)).item()
                        for t in range(len(x0_estimates))]
                mse_clean = [(torch.mean((x0_estimates_clean[t].cpu() - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]

                mse_ancestor_all[name].append(mse_ancestor)
                mse_clean_all[name].append(mse_clean)

    # 3. Aggregate results
   
    mean_mse_ancestor = {}
    mean_mse_clean = {}
    for name in models:
        # Concatenate all batches along dimension 0
        mean_mse_ancestor[name] = torch.mean(torch.cat(mse_ancestor_all[name], dim=0), dim=0)
        mean_mse_clean[name] = torch.mean(torch.cat(mse_clean_all[name], dim=0), dim=0)

    return {
        "mse_ancestor": mean_mse_ancestor,   # (N_total, T, C, H, W)
        "mse_clean": mean_mse_clean,   # (N_total, T, C, H, W)
    }




def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models on Turbulence Data")
    
    # Paths
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset root")
    
    # UPDATED ARGUMENT:
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                        help="List of checkpoints. Format: 'Name=/path/to/ckpt.pth'. Space separated.")
    
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots and metrics")
    
    # Params
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=50) 
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
        "data_path": args.data_path,
        "dataset_name": "KolmogorovFlow",
        "resolution": args.resolution,
        "sequence_length": [2, 1],
        "trajectory_sequence_length": [64, 1], 
        "frames_per_time_step": 1,
        "limit_trajectories_train": 100,
        "limit_trajectories_val": args.batch_size, 
        "batch_size": args.batch_size
    }
    _, val_loader, _ = get_data_loaders(data_params)

    # 2. Load Candidate Models (Looping over Dictionary)
    models = {}
    for name, ckpt_path in checkpoints_dict.items():
        print(f"--- Loading Candidate: {name} ---")
        
        # Initialize Architecture
        model = DiffusionModel(
            dimension=2,
            dataSize=[64, 64],
            condChannels=2,
            diffSchedule="psd",
            diffSteps=100,
            inferenceSamplingMode="ddpm",
            inferenceConditioningIntegration="clean",
            diffCondIntegration="clean",
            inferenceInitialSampling="random",
            x0_estimate_type="mean"
        ).to(args.device)
        
        # Load weights
        ckpt = torch.load(ckpt_path, map_location=args.device)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
            
        models[name] = model

    # 3. Evaluate train input vs inference input predictions
    results = evaluate_dw_train_inf_gap(models, val_loader, device='cuda')

    
        