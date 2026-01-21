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
from src.dataset import TurbulenceDataset
from src.data_transformations import DataParams, Transforms
from src.model_diffusion import DiffusionModel
from src.model import Unet
from src.utils import count_parameters, parse_checkpoint_args, run_model, run_model
from torch.utils.data import DataLoader, SequentialSampler
from src.diffusion_utils import betas_from_sqrtOneMinusAlphasCumprod


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
        mse_clean_own_pred_all = {name: [] for name in models}
        mse_clean_prev_pred_all = {name: [] for name in models}

        for batch_idx, sample in enumerate(val_loader):
            
            # --- A. Prepare Batch ---
            data = sample["data"].to(device) # (B, T_total, C, H, W)
            batch_size = data.shape[0]
            total_samples += batch_size

            # Initial Condition (t=0)
            conditioning_frame = torch.concatenate((data[:, 0], data[:, 1]), dim=1)
            target_frame = data[:, 2]

            for name in models:
                print('a')
                # Store prediction
                model = models[name]
                _, x0_estimates = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="ancestor")
                _, x0_estimates_clean = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="clean")

                _, x0_estimates_clean_own_pred = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="own-pred")
                #_, x0_estimates_clean_prev_pred = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="prev-pred")

                
                mse_ancestor = [(torch.mean((x0_estimates[t] - target_frame)**2)).item()
                        for t in range(len(x0_estimates))]
                mse_clean = [(torch.mean((x0_estimates_clean[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]
            
                mse_clean_own_pred = [(torch.mean((x0_estimates_clean_own_pred[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]
                
                #mse_clean_prev_pred = [(torch.mean((x0_estimates_clean_prev_pred[t] - target_frame)**2)).item()
                #         for t in range(len(x0_estimates))]
                
                
                mse_ancestor_all[name].append(mse_ancestor)
                mse_clean_all[name].append(mse_clean)
                mse_clean_own_pred_all[name].append(mse_clean_own_pred)
                #mse_clean_prev_pred_all[name].append(mse_clean_prev_pred)

            if batch_idx == 0:
                break

    # 3. Aggregate results
   
    mean_mse_ancestor = {}
    mean_mse_clean = {}
    mean_mse_clean_own_pred = {}
    mean_mse_clean_prev_pred = {}
    for name in models:
        # Concatenate all batches along dimension 0
        mean_mse_ancestor[name] = torch.mean(torch.tensor(mse_ancestor_all[name]), dim=0)
        mean_mse_clean[name] = torch.mean(torch.tensor(mse_clean_all[name]), dim=0)
        mean_mse_clean_own_pred[name] = torch.mean(torch.tensor(mse_clean_own_pred_all[name]), dim=0)
        #mean_mse_clean_prev_pred[name] = torch.mean(torch.tensor(mse_clean_prev_pred_all[name]), dim=0)

    return {
        "mse_ancestor": mean_mse_ancestor,   # (N_total, T, C, H, W)
        "mse_clean": mean_mse_clean,   # (N_total, T, C, H, W)
        "mse_clean_own_pred": mean_mse_clean_own_pred,   # (N_total, T, C, H, W)
        #"mse_clean_prev_pred": mean_mse_clean_prev_pred,   # (N_total, T, C, H, W)
    }



def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models on Turbulence Data")
    
    # Paths
    parser.add_argument('--data_path', type=str, default="/mnt/SSD2/constantin/autoreg-pde-diffusion/data", help="Path to dataset root")
    
    # UPDATED ARGUMENT:
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                        help="List of checkpoints. Format: 'Name=/path/to/ckpt.pth'. Space separated.")
    
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots and metrics")
    
    # Params
    parser.add_argument('--resolution', type=int, default=64)
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
    p_d_test = DataParams(batch=100, augmentations=["normalize"], sequenceLength=[(3,2)], randSeqOffset=False,
            dataSize=[128,64], dimension=2, simFields=["dens", "pres"], simParams=["mach"], normalizeMode="traMixed")
    testSet = TurbulenceDataset("Training", [args.data_path], filterTop=["128_tra"], filterSim=[[0,1,2,14,15,16,17,18]], excludefilterSim=True, filterFrame=[(0,1000)],
                        sequenceLength=p_d_test.sequenceLength, randSeqOffset=p_d_test.randSeqOffset, simFields=p_d_test.simFields, simParams=p_d_test.simParams, printLevel="sim")

    transTest = Transforms(p_d_test)
    testSet.transform = transTest
    testSampler = SequentialSampler(testSet)
    testLoader = DataLoader(testSet, sampler=testSampler, batch_size=p_d_test.batch, drop_last=True, num_workers=4)

    condChannels =  (2 + len(p_d_test.simFields) + len(p_d_test.simParams))
    dataChannels = 2 + len(p_d_test.simFields) + len(p_d_test.simParams)
    print(condChannels, dataChannels)

    # 2. Load Candidate Models (Looping over Dictionary)
    models = {}
    for name, ckpt_path in checkpoints_dict.items():
        print(f"--- Loading Candidate: {name} ---")
        
        # Initialize Architecture
        model = DiffusionModel(
            dimension=2,
            dataSize=[128, 64],
            condChannels=2*condChannels,
            dataChannels=dataChannels,
            diffSchedule="transonicIteration7",
            diffSteps=100,
            inferenceSamplingMode="ddpm",
            inferenceConditioningIntegration="clean",
            diffCondIntegration="clean",
            inferenceInitialSampling="random",
            architecture="Unet2D"
        ).to(args.device)
        
        # Load weights
        ckpt = torch.load(ckpt_path, map_location=args.device)['stateDictDecoder']
        print(ckpt.keys())
        ckpt = {key[5:]:ckpt[key] for key in ckpt if 'unet' in key and not 'sigmas' in key}
        print(ckpt)
        model.unet.load_state_dict(ckpt)
        """if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        """
        models[name] = model

    # 3. Evaluate train input vs inference input predictions
    results = evaluate_dw_train_inf_gap(models, testLoader, device='cuda')

    n_models = len(checkpoints_dict.items())
    fig, axes = plt.subplots(2, n_models, figsize=(3*len(models), 6), sharex=True)

    colors = ['purple', 'blue', 'red']

    # --------------------
    for i, model_name in enumerate(models):
        if n_models == 1:
            ax_model = axes
        else:
            ax_model = axes[:, i]

        model = models[model_name]
        mse_ancestor = results['mse_ancestor'][model_name]
        mse_clean = results['mse_clean'][model_name]
        mse_clean_own_pred = results['mse_clean_own_pred'][model_name]
        alphas = list(models[model_name].sqrtOneMinusAlphasCumprod.ravel().cpu())[::-1]

        ax_model[0].hist(alphas, alpha=0.2, color=colors[i], bins=np.logspace(-2.5, 0, 20))

        tau = 1.05

        # Plot curves
        ax_model[1].plot(alphas, mse_clean,
                    label="Training input", color=colors[i], linestyle='dotted')
        ax_model[1].plot(alphas, mse_ancestor,
                    label="Inference input", color=colors[i])
        ax_model[1].plot(alphas, mse_clean_own_pred,
                    label="Inference input", color=colors[i], linestyle='dashdot')
        # Grid and title
        ax_model[1].grid(True, which='both', linestyle='--', alpha=0.3)
        ax_model[0].set_title(model_name)
        print("\n")
        fig.text(
            0.2 + i*0.33,   # places 3 groups left→right
            0,            # slightly below suptitle
            f"Training Final Error:  {mse_clean[-1]:.2e}\n"
            f"Inference Final Error: {mse_ancestor[-1]:.2e}",
            ha='center',
            va='top',
            fontsize=8,
            color=colors[i]
        )
        ax_model[1].legend(fontsize=8)

    axes[1].sharex(axes[0])
    axes[0].set_xscale('log')
    axes[1].set_yscale('log')

        
    axes[1].set_ylabel('MSE w/ ground-truth')
    axes[0].set_ylabel('Noise Level sqrt(1-ᾱₜ) Distribution')
    axes[1].set_xlabel('Noise Level sqrt(1-ᾱₜ), High = Only Noise, Low = Only Image')

    plt.savefig(os.path.join(args.output_dir, "train-inf-gap.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()