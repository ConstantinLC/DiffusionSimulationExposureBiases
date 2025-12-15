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
from src.utils import fsd_torch_radial
from torch.utils.data import DataLoader, SequentialSampler


# ==========================================
# 3. Model Evaluation Logic
# ==========================================

def evaluate_rollout(models, traj_loader, device, rollout_steps=30):
    """
    Runs autoregressive rollout for multiple models over the FULL dataset.
    """
    # 1. Set models to eval mode
    for model in list(models.values()):
        model.eval()

    # 2. Initialize accumulators
    # We will store predictions on CPU to avoid GPU OOM
    all_predictions = {name: [] for name in models} 
    all_ground_truth = []
    
    # We will track the running sum of distances to average them later
    # Shape: (rollout_steps, )
    running_eval_distances = {name: torch.zeros(rollout_steps, device=device) for name in models}
    total_samples = 0

    print(f"Starting evaluation over full dataset ({len(traj_loader)} batches)...")

    with torch.no_grad():
        for batch_idx, sample in enumerate(traj_loader):
            
            # --- A. Prepare Batch ---
            data = sample["data"].to(device) # (B, T_total, C, H, W)
            batch_size = data.shape[0]
            total_samples += batch_size
            
            # Store Ground Truth (CPU)
            all_ground_truth.append(data.cpu())

            # Initial Condition (t=0)
            conditioning_frame = torch.cat((data[:, 0], data[:, 1]), dim=1)
            # Current state buffer for this batch
            current_inputs = {name: conditioning_frame for name, model in models.items()}
            current_preds = {name: run_model(model, conditioning_frame) for name, model in models.items()}
            
                        # Batch trajectory buffer: List of T tensors, each (B, C, H, W)
            batch_trajectory_buffer = {name: [] for name in models}
            
            for name in models:
                # Store prediction
                batch_trajectory_buffer[name].append(current_preds[name])

            # --- C. Autoregressive Rollout (Steps 1 to T) ---
            for t in range(1, rollout_steps):
                for name, model in models.items():
                    # 1. Get evaluator score on *current* input (before stepping)
                    # Note: In your previous logic, you compared Evaluator(x_t) vs x_{t+1}.
                    # If you want Evaluator(x_t) vs Model(x_t), align indices carefully. 
                    # Below follows your original logic: eval_pred = m_eval(current), next = model(current)
                    current_inputs[name] = torch.cat((current_inputs[name][:, -model.dataChannels:], current_preds[name]), dim=1)
                    # 2. Step forward
                    current_preds[name] = run_model(model, current_inputs[name])
                    batch_trajectory_buffer[name].append(current_preds[name])
                    

            # --- D. Store Batch Results to CPU ---
            for name in models:
                # Stack time dimension: (B, T, C, H, W)
                full_batch_traj = torch.stack(batch_trajectory_buffer[name], dim=1)
                all_predictions[name].append(full_batch_traj.cpu())

            # Optional: Print progress
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(traj_loader)}")

    # 3. Aggregate Results
    print("Aggregating results...")
    
    final_predictions = {}
    for name in models:
        # Concatenate all batches along dimension 0
        final_predictions[name] = torch.cat(all_predictions[name], dim=0)
        print(final_predictions[name].shape)
        
    final_ground_truth = torch.cat(all_ground_truth, dim=0)

    print(f"Evaluation Complete. Total samples: {total_samples}")
    print(f"Prediction Shape: {final_predictions[list(models.keys())[0]].shape}")

    
    print(final_ground_truth.shape)
    return {
        "predictions": final_predictions,   # (N_total, T, C, H, W)
        "data": final_ground_truth          # (N_total, T_total, C, H, W)
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
    parser.add_argument('--rollout_steps', type=int, default=20)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 0. Parse Checkpoints into Dictionary
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)
    print(f"--- Evaluating {len(checkpoints_dict)} Models ---")
    for name, path in checkpoints_dict.items():
        print(f"  > {name}: {path}")

    # 1. Load Data
    p_d_test = DataParams(batch=4, augmentations=["normalize"], sequenceLength=[(60,2)], randSeqOffset=False,
            dataSize=[128,64], dimension=2, simFields=["dens", "pres"], simParams=["mach"], normalizeMode="traMixed")

    #testSet = TurbulenceDataset("Test Interpolate Mach 0.66-0.68", [args.data_path], filterTop=["128_tra"], filterSim=[(16,19)],
    #                filterFrame=[(500,750)], sequenceLength=p_d_test.sequenceLength, simFields=p_d_test.simFields, simParams=p_d_test.simParams, printLevel="sim")
    

    testSet = TurbulenceDataset("Test Extrapolate Mach 0.50-0.52", [args.data_path], filterTop=["128_tra"], filterSim=[(0,3)],
                        filterFrame=[(500,750)], sequenceLength=[[60,2]], simFields=p_d_test.simFields, simParams=p_d_test.simParams, printLevel="sim")

    print(len(testSet))
    transTest = Transforms(p_d_test)
    testSet.transform = transTest
    testSampler = SequentialSampler(testSet)
    traj_loader = DataLoader(testSet, sampler=testSampler, batch_size=p_d_test.batch, drop_last=True, num_workers=4)

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

    # 4. Run Evaluation
    print("--- Running Rollouts ---")
    results = evaluate_rollout(models, traj_loader, args.device, args.rollout_steps)

    # 5. Compute Metrics & Plotting
    print("--- Computing Metrics ---")
    
    final_metrics = {}

    gt_trajectory = results["data"][:, 2:args.rollout_steps] 
    

    # Setup Figures
    fig_mse, ax_mse = plt.subplots(figsize=(10, 6))
    fig_fsd, ax_fsd = plt.subplots(figsize=(10, 6))
    fig_eval, ax_eval = plt.subplots(figsize=(10, 6))

    for name in models:
        preds = results["predictions"][name] 
        preds = preds[:, :-2]
        
        # A. MSE
        mse_time = torch.mean((preds - gt_trajectory)**2, dim=(0,2,3,4)).cpu().numpy()
        ax_mse.plot(mse_time, label=name)
        
        # B. FSD
        fsd_time = []
        for t in range(args.rollout_steps-2):
            val = fsd_torch_radial(preds[:, t], gt_trajectory[:, t])
            fsd_time.append(val.item())
        ax_fsd.plot(fsd_time, label=name)
        
        final_metrics[name] = {
            "final FSD": float(fsd_time[-1]),
            "Step 1 MSE": float(mse_time[0]),
            "Step 10 MSE": float(mse_time[10]),
            "Last step MSE": float(mse_time[-1])
        }

    # Finalize Plots
    ax_mse.set_yscale('log')
    ax_mse.set_title("MSE vs Time Step")
    ax_mse.set_xlabel("Time Step")
    ax_mse.set_ylabel("MSE")
    ax_mse.legend()
    fig_mse.savefig(os.path.join(args.output_dir, "metric_mse.png"))
    
    ax_fsd.set_yscale('log')
    ax_fsd.set_title("FSD (Radial Spectrum) vs Time Step")
    ax_fsd.set_xlabel("Time Step")
    ax_fsd.set_ylabel("FSD")
    ax_fsd.legend()
    fig_fsd.savefig(os.path.join(args.output_dir, "metric_fsd.png"))

    ax_eval.set_yscale('log')
    ax_eval.set_title("Distance to Evaluator Model vs Time Step")
    ax_eval.set_xlabel("Time Step")
    ax_eval.legend()
    fig_eval.savefig(os.path.join(args.output_dir, "metric_evaluator_dist.png"))
    
    # Save Scalar Metrics
    with open(os.path.join(args.output_dir, "metrics_summary.json"), 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()