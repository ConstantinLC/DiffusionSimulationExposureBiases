#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler

import sys
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

# --- Project Imports ---
# Assuming these exist in your src folder as per the previous snippets
from src.dataset import TurbulenceDataset
from src.data_transformations import DataParams, Transforms
from src.model_diffusion import DiffusionModel

def evaluate_dw_train_inf_gap(models, val_loader, device):
    """
    Runs autoregressive rollout for multiple models over the FULL dataset.
    """
    # 1. Set models to eval mode
    for model in list(models.values()):
        model.eval()
    
    print(f"Starting evaluation over full dataset ({len(val_loader)} batches)...")

    with torch.no_grad():
        mse_ancestor_all = {name: [] for name in models}
        mse_clean_all = {name: [] for name in models}
        mse_clean_own_pred_all = {name: [] for name in models}

        for batch_idx, sample in enumerate(val_loader):
            # --- A. Prepare Batch ---
            data = sample["data"].to(device) # (B, T_total, C, H, W)
            
            # Initial Condition (t=0) and Target (t=1)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]

            for name in models:
                model = models[name]
                
                # 1. Ancestor Sampling (Inference)
                _, x0_estimates = model(conditioning=conditioning_frame, data=target_frame, 
                                      return_x0_estimate=True, input_type="ancestor")
                
                # 2. Clean Input (Teacher Forcing)
                _, x0_estimates_clean = model(conditioning=conditioning_frame, data=target_frame, 
                                            return_x0_estimate=True, input_type="clean")

                # 3. Own Prediction (One-step rollout)
                _, x0_estimates_own = model(conditioning=conditioning_frame, data=target_frame, 
                                          return_x0_estimate=True, input_type="own-pred")

                # Calculate MSE per timestep
                mse_ancestor = [(torch.mean((est - target_frame)**2)).item() for est in x0_estimates]
                mse_clean = [(torch.mean((est - target_frame)**2)).item() for est in x0_estimates_clean]
                mse_own = [(torch.mean((est - target_frame)**2)).item() for est in x0_estimates_own]
                
                mse_ancestor_all[name].append(mse_ancestor)
                mse_clean_all[name].append(mse_clean)
                mse_clean_own_pred_all[name].append(mse_own)

            # Optional: Limit evaluation to first batch for speed if needed
            if batch_idx == 0:
                break

    # 3. Aggregate results
    results = {
        "mse_ancestor": {},
        "mse_clean": {},
        "mse_clean_own_pred": {}
    }
    
    for name in models:
        results["mse_ancestor"][name] = torch.mean(torch.tensor(mse_ancestor_all[name]), dim=0)
        results["mse_clean"][name] = torch.mean(torch.tensor(mse_clean_all[name]), dim=0)
        results["mse_clean_own_pred"][name] = torch.mean(torch.tensor(mse_clean_own_pred_all[name]), dim=0)

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models Rounds")
    
    # Paths
    parser.add_argument('--data_path', type=str, default="/mnt/SSD2/constantin/autoreg-pde-diffusion/data", help="Path to dataset root")
    parser.add_argument('--base_experiment_dir', type=str, required=True, 
                        help="Path to the base run directory (e.g., checkpoints/run_659)")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================================
    # 1. Find Rounds and Load Configs
    # ==========================================
    base_dir = args.base_experiment_dir
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory {base_dir} does not exist.")

    # Find all 'round_X' directories
    rounds = [d for d in os.listdir(base_dir) if d.startswith("round_") and os.path.isdir(os.path.join(base_dir, d))]
    # Sort rounds numerically (round_1, round_2, ...)
    rounds = sorted(rounds, key=lambda x: int(x.split('_')[1]))

    print(f"--- Found {len(rounds)} Rounds in {base_dir} ---")
    
    # ==========================================
    # 2. Data Loading (Initialize once)
    # ==========================================
    p_d_test = DataParams(batch=100, augmentations=["normalize"], sequenceLength=[(2,2)], randSeqOffset=False,
            dataSize=[128,64], dimension=2, simFields=["dens", "pres"], simParams=["mach"], normalizeMode="traMixed")
    
    testSet = TurbulenceDataset("Training", [args.data_path], filterTop=["128_tra"], filterSim=[[0,1,2,14,15,16,17,18]], 
                                excludefilterSim=True, filterFrame=[(0,1000)],
                                sequenceLength=p_d_test.sequenceLength, randSeqOffset=p_d_test.randSeqOffset, 
                                simFields=p_d_test.simFields, simParams=p_d_test.simParams, printLevel="sim")

    transTest = Transforms(p_d_test)
    testSet.transform = transTest
    testSampler = SequentialSampler(testSet)
    testLoader = DataLoader(testSet, sampler=testSampler, batch_size=p_d_test.batch, drop_last=True, num_workers=4)

    condChannels = (2 + len(p_d_test.simFields) + len(p_d_test.simParams))
    dataChannels = 2 + len(p_d_test.simFields) + len(p_d_test.simParams)

    # ==========================================
    # 3. Load Models per Round
    # ==========================================
    models = {}
    
    for r_name in rounds:
        round_path = os.path.join(base_dir, r_name)
        ckpt_path = os.path.join(round_path, "best_model.pth")
        schedule_path = os.path.join(round_path, "schedule_params.json")

        if not os.path.exists(ckpt_path):
            print(f"Skipping {r_name}: best_model.pth not found.")
            continue
        
        print(f"--- Loading {r_name} ---")
        
        # A. Load Schedule Params
        noise_levels_tensor = None
        if os.path.exists(schedule_path):
            with open(schedule_path, 'r') as f:
                sched_data = json.load(f)
                # Handle dictionary vs list format
                if isinstance(sched_data, dict) and "noise_levels" in sched_data:
                     nl_list = sched_data["noise_levels"]
                elif isinstance(sched_data, list):
                     nl_list = sched_data
                else:
                    print(f"Warning: Could not parse schedule for {r_name}, using default linear.")
                    nl_list = None
                
                if nl_list is not None:
                    noise_levels_tensor = torch.tensor(nl_list, device=args.device, dtype=torch.float32)
        
        # Determine steps from loaded schedule or default
        steps = len(noise_levels_tensor) if noise_levels_tensor is not None else 100

        # B. Initialize Model Architecture
        model = DiffusionModel(
            dimension=2,
            dataSize=[128, 64],
            condChannels=condChannels,
            dataChannels=dataChannels,
            diffSchedule="linear", # Placeholder, will overwrite if custom schedule exists
            diffSteps=steps, 
            inferenceSamplingMode="ddpm",
            inferenceConditioningIntegration="clean",
            diffCondIntegration="clean",
            inferenceInitialSampling="random",
            architecture="ACDM"
        ).to(args.device)

        # C. Overwrite Schedule with Loaded Params
        if noise_levels_tensor is not None:
            print(f"  > Overwriting schedule with {steps} steps from json.")
            model.compute_schedule_variables(sigmas=noise_levels_tensor)

        # D. Load Weights
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        
        models[r_name] = model

    if not models:
        print("No models loaded. Exiting.")
        return

    # ==========================================
    # 4. Evaluate
    # ==========================================
    results = evaluate_dw_train_inf_gap(models, testLoader, device=args.device)

    # ==========================================
    # 5. Plotting
    # ==========================================
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8), sharex=False, sharey='row')
    # If n_models is 1, axes is 1D array or scalar depending on mpl version, ensure 2D indexing works
    if n_models == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif axes.ndim == 1: 
        # Fallback if reshape needed
        axes = axes.reshape(2, n_models)

    colors = ['#6a0dad', '#1f77b4', '#d62728'] # Purple, Blue, Red

    # Sort models by round number for plotting order
    sorted_names = sorted(models.keys(), key=lambda x: int(x.split('_')[1]))

    for i, name in enumerate(sorted_names):
        model = models[name]
        
        # Get data
        mse_clean = results['mse_clean'][name]
        mse_ancestor = results['mse_ancestor'][name]
        mse_own = results['mse_clean_own_pred'][name]
        
        # X-axis: Noise Levels (sqrt(1-alpha_bar)). 
        # Note: Model stores them from 0 (low noise) to T (high noise) usually, 
        # or vice versa depending on implementation. 
        # Standard DDPM: T is pure noise. 
        alphas = model.sqrtOneMinusAlphasCumprod.ravel().cpu().numpy()
        
        # Plot Top: Distribution of schedule points
        ax_top = axes[0, i]
        ax_top.hist(alphas, alpha=0.3, color=colors[0], bins=np.logspace(np.log10(min(alphas)+1e-9), 0, 20))
        ax_top.set_xscale('log')
        ax_top.set_title(name)
        
        # Plot Bottom: MSE Curves
        ax_bot = axes[1, i]
        ax_bot.plot(alphas, torch.flip(mse_clean, [0]), label="Train Input (Clean)", color=colors[0], linestyle=':')
        ax_bot.plot(alphas, torch.flip(mse_ancestor, [0]), label="Inference (Ancestor)", color=colors[1])
        ax_bot.plot(alphas, torch.flip(mse_own, [0]), label="Own Pred", color=colors[2], linestyle='-.')
        
        ax_bot.set_xscale('log')
        ax_bot.set_yscale('log')
        ax_bot.grid(True, which='both', linestyle='--', alpha=0.3)
        
        # Text Metrics
        fig.text(
            (i + 0.5) / n_models, 0.02, 
            f"Train Final: {mse_clean[-1]:.2e}\nInfer Final: {mse_ancestor[-1]:.2e}",
            ha='center', fontsize=9, color='black', transform=fig.transFigure
        )
        
        if i == 0:
            ax_top.set_ylabel('Schedule Density')
            ax_bot.set_ylabel('MSE w/ Ground Truth')
            ax_bot.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # Make room for text
    save_path = os.path.join(args.output_dir, f"rounds_comparison_{os.path.basename(base_dir)}.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()