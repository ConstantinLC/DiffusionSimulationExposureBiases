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
        mse_clean_prev_all = {name: [] for name in models}

        for batch_idx, sample in enumerate(val_loader):
            
            # --- A. Prepare Batch ---
            data = sample["data"].to(device) # (B, T_total, C, H, W)
            
            batch_size = data.shape[0]
            total_samples += batch_size

            # Initial Condition (t=0)
            conditioning_frame = torch.cat((data[:, 0], data[:, 1]), dim=1)
            target_frame = data[:, 2]

            for name in models:
                # Store prediction
                model = models[name]
                print(model.sqrtAlphasCumprod.ravel())
                _, x0_estimates = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="ancestor")
                _, x0_estimates_clean = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="clean")
                _, x0_estimates_clean_previous = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="clean-previous-1")

                mse_ancestor = [(torch.mean((x0_estimates[t] - target_frame)**2)).item()
                        for t in range(len(x0_estimates))]
                mse_clean = [(torch.mean((x0_estimates_clean[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]
                mse_clean_prev = [(torch.mean((x0_estimates_clean_previous[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates_clean_previous))]

                mse_ancestor_all[name].append(mse_ancestor)
                mse_clean_all[name].append(mse_clean)
                mse_clean_prev_all[name].append(mse_clean_prev)
            
            break

    # 3. Aggregate results
   
    mean_mse_ancestor = {}
    mean_mse_clean = {}
    mean_mse_clean_prev = {}

    for name in models:
        # Concatenate all batches along dimension 0
        mean_mse_ancestor[name] = torch.mean(torch.tensor(mse_ancestor_all[name]), dim=0)
        mean_mse_clean[name] = torch.mean(torch.tensor(mse_clean_all[name]), dim=0)
        mean_mse_clean_prev[name] = torch.mean(torch.tensor(mse_clean_prev_all[name]), dim=0)

    return {
        "mse_ancestor": mean_mse_ancestor,   # (N_total, T, C, H, W)
        "mse_clean": mean_mse_clean,   # (N_total, T, C, H, W)
        "mse_clean_prev": mean_mse_clean_prev,   # (N_total, T, C, H, W)
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
    parser.add_argument('--limit_val_trajectories', type=int, default=10) 
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

    condChannels = 2 * (2 + len(p_d_test.simFields) + len(p_d_test.simParams))
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
            condChannels=condChannels,
            dataChannels=dataChannels,
            diffSchedule="linear",
            diffSteps=20,
            inferenceSamplingMode="ddpm",
            inferenceConditioningIntegration="clean",
            diffCondIntegration="clean",
            inferenceInitialSampling="random",
            x0_estimate_type="mean",
            architecture="ACDM"
        ).to(args.device)
        
        # Load weights
        ckpt = torch.load(ckpt_path, map_location=args.device)['stateDictDecoder']
        print(ckpt.keys())
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
            
        models[name] = model

    # 3. Evaluate train input vs inference input predictions
    results = evaluate_dw_train_inf_gap(models, testLoader, device='cuda')

    n_models = len(checkpoints_dict.items())
    fig, axes = plt.subplots(2, n_models, figsize=(3*len(models), 6), sharex=True)

    colors = ['purple', 'blue', 'red']

    # --------------------
    for i, model_name in enumerate(models):
        mse_ancestor = results['mse_ancestor'][model_name]
        mse_clean = results['mse_clean'][model_name]
        mse_clean_prev = results['mse_clean_prev'][model_name]
        alphas = list(models[model_name].sqrtOneMinusAlphasCumprod.ravel().cpu())[::-1]

        axes[0,i].hist(alphas, alpha=0.2, color=colors[i], bins=np.logspace(-2.5, 0, 20))

        # Plot curves
        axes[1,i].plot(alphas, mse_clean,
                    label="Training input", color=colors[i], linestyle='dotted')
        axes[1,i].plot(alphas, mse_ancestor,
                    label="Inference input", color=colors[i])
        axes[1,i].plot(alphas[1:], mse_clean_prev[1:],
                    label="Training input on previous step", color="green")

        # Grid and title
        axes[1,i].grid(True, which='both', linestyle='--', alpha=0.3)
        axes[0,i].set_title(model_name)

        # --- Add final-value text under the title ---
        #print(model_name, "Clean:", mse_clean[0], "Ancestor:", mse_ancestor[0])
        print([(mse_ancestor[i]/mse_clean_prev[i]) for i in range(len(mse_ancestor))])
        print(model_name, "Clean:", mse_clean[-1], "Ancestor:", mse_ancestor[-1], "Clean on previous step:", mse_clean_prev[-1])
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

        axes[1,i].sharey(axes[1,-1])
        axes[1,i].legend(fontsize=8)
        

    axes[0,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    axes[1,0].set_ylabel('MSE w/ ground-truth')
    axes[0,0].set_ylabel('Noise Level sqrt(1-ᾱₜ) Distribution')
    axes[1,1].set_xlabel('Noise Level sqrt(1-ᾱₜ), High = Only Noise, Low = Only Image')

    plt.savefig(os.path.join(args.output_dir, "train-inf-gap.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()