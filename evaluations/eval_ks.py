#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch import nn
import random
import sys

sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

# --- Project Imports ---
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.model_1d import Unet1D
from src.utils import count_parameters, parse_checkpoint_args, run_model, run_model
from src.utils import correlation, vorticity
#from src.utils import evaluate_trajectory_vorticity

# --- 0. Determinism Utils ---
def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may slow down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Seed set to {seed} ---")


def evaluate_trajectory(predictions, ground_truth, threshold=0.8):
    N, T, C, H = predictions.shape
        
    corrs = np.zeros((N, T))
    
    for n in range(N):
        for t in range(T):
            corrs[n, t] = correlation(predictions[n, t], ground_truth[n, t])
            
    # --- Aggregates ---
    mean_correlations = np.mean(corrs, axis=0) 
    std_per_timestep = np.std(corrs, axis=0)

    # --- Time Until Failure ---
    below_threshold_mask = corrs < threshold
    first_failure_indices = np.argmax(below_threshold_mask, axis=1)
    has_failed = np.any(below_threshold_mask, axis=1)
    times_under_threshold = np.where(has_failed, first_failure_indices + 1, T)
    
    sorted_times = np.sort(times_under_threshold)
    cutoff_idx = max(1, int(0.1 * N)) 
    worst_10_mean = np.mean(sorted_times[:cutoff_idx])
    top_10_mean = np.mean(sorted_times[-cutoff_idx:])

    return {
        'mean_correlations': mean_correlations,
        'std_per_timestep': std_per_timestep,
        'time_under_threshold': np.mean(times_under_threshold),
        'time_under_threshold_worst_10': worst_10_mean,
        'time_under_threshold_best_10': top_10_mean,
    }


# ==========================================
# 3. Model Evaluation Logic
# ==========================================

def evaluate_rollout_with_evaluator(models, m_eval, traj_loader, device, rollout_steps=30):
    """
    Runs autoregressive rollout for multiple models over the FULL dataset.
    """
    # 1. Set models to eval mode
    for model in list(models.values()) + [m_eval]:
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
            conditioning_frame = data[:, 0]
            
            # Current state buffer for this batch
            current_preds = {name: run_model(model, conditioning_frame) for name, model in models.items()}
            
            # Batch trajectory buffer: List of T tensors, each (B, C, H, W)
            batch_trajectory_buffer = {name: [] for name in models}

            # --- B. Process Step 0 (Initial Prediction) ---
            # We calculate this outside the loop to handle the "0-th" step logic cleanly
            pred_eval_t0 = run_model(m_eval, conditioning_frame)
            
            for name in models:
                # Store prediction
                batch_trajectory_buffer[name].append(current_preds[name])
                
                # Compute distance vs Evaluator
                # Mean over pixels/channels (C,H,W), Sum over Batch (B)
                dist = torch.sum((pred_eval_t0 - current_preds[name]) ** 2, dim=(1,2))
                running_eval_distances[name][0] += dist.sum()

            # --- C. Autoregressive Rollout (Steps 1 to T) ---
            for t in range(1, rollout_steps):
                for name, model in models.items():
                    # 1. Get evaluator score on *current* input (before stepping)
                    # Note: In your previous logic, you compared Evaluator(x_t) vs x_{t+1}.
                    # If you want Evaluator(x_t) vs Model(x_t), align indices carefully. 
                    # Below follows your original logic: eval_pred = m_eval(current), next = model(current)
                    eval_pred_on_model = run_model(m_eval, current_preds[name])
                    
                    # 2. Step forward
                    current_preds[name] = run_model(model, current_preds[name])
                    batch_trajectory_buffer[name].append(current_preds[name])
                    
                    # 3. Compute distance
                    # Squared diff between "Evaluator prediction" and "Model prediction"
                    diff = (eval_pred_on_model - current_preds[name]) ** 2
                    dist_sum = torch.sum(diff, dim=(1,2)).sum() # Sum over batch
                    running_eval_distances[name][t] += dist_sum

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
        
    final_ground_truth = torch.cat(all_ground_truth, dim=0)

    # Average the distances
    final_eval_distances = {}
    for name in models:
        # Divide sum by total samples to get Mean Squared Error
        avg_dist_tensor = running_eval_distances[name] / total_samples
        # Convert to list of scalars to match original format
        final_eval_distances[name] = [avg_dist_tensor[t] for t in range(rollout_steps)]

    print(f"Evaluation Complete. Total samples: {total_samples}")
    print(f"Prediction Shape: {final_predictions[list(models.keys())[0]].shape}")

    return {
        "predictions": final_predictions,   # (N_total, T, C, H, W)
        "eval_distances": final_eval_distances, # Dict of List of scalars
        "data": final_ground_truth          # (N_total, T_total, C, H, W)
    }


def evaluate_rollout_alone(models, traj_loader, device, rollout_steps=30):
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
            conditioning_frame = data[:, 0]
            
            # Current state buffer for this batch
            current_preds = {name: run_model(model, conditioning_frame) for name, model in models.items()}

            for name in models:
                print(torch.mean((current_preds[name] - data[:, 1])**2))
            
            # Batch trajectory buffer: List of T tensors, each (B, C, H, W)
            batch_trajectory_buffer = {name: [] for name in models}

            # --- B. Process Step 0 (Initial Prediction) ---
            # We calculate this outside the loop to handle the "0-th" step logic cleanly            
            for name in models:
                # Store prediction
                batch_trajectory_buffer[name].append(current_preds[name])
            for t in range(1, rollout_steps):
                for name, model in models.items():
                    current_preds[name] = run_model(model, current_preds[name])
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
        
    final_ground_truth = torch.cat(all_ground_truth, dim=0)

    print(f"Evaluation Complete. Total samples: {total_samples}")
    print(f"Prediction Shape: {final_predictions[list(models.keys())[0]].shape}")

    return {
        "predictions": final_predictions,   # (N_total, T, C, H, W)
        "data": final_ground_truth          # (N_total, T_total, C, H, W)
    }

# ==========================================
# 4. Main Script
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models on Turbulence Data")
    
    # Paths
    parser.add_argument('--eval_model_path', type=str, default=None, help="Path to the pretrained Evaluator UNet (.pth)")
    
    # UPDATED ARGUMENT:
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                        help="List of checkpoints. Format: 'Name=/path/to/ckpt.pth'. Space separated.")
    
    
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots and metrics")

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--rollout_steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, 'r') as f: config = json.load(f)

    set_seed(args.seed)
    
    # 0. Parse Checkpoints into Dictionary
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)
        
    print(f"--- Evaluating {len(checkpoints_dict)} Models ---")
    for name, path in checkpoints_dict.items():
        print(f"  > {name}: {path}")

    # 1. Load Data
    _, _, traj_loader = get_data_loaders(config['data_params'])

    # 2. Load Evaluator Model
    
    print(f"--- Loading Evaluator Model: {args.eval_model_path} ---")
    m_eval = Unet1D(
        sigmas=torch.tensor([1]),
        dim=64, channels=1, dim_mults=(1,1,1), 
        use_convnext=True, convnext_mult=1, with_time_emb=False
    ).to(args.device)

    m_eval.eval()

    # 3. Load Candidate Models (Looping over Dictionary)
    models = {}
    for name, ckpt_path in checkpoints_dict.items():
        print(f"--- Loading Candidate: {name} ---")
        
        if "Diffusion" in name:
            model_config = config['model_params']
            model_config['checkpoint'] = checkpoints_dict[name]
            model_config['load_betas'] = True
            
            model = DiffusionModel(**model_config).to(args.device)

            #model.compute_schedule_variables(sigmas = model.sqrtOneMinusAlphasCumprod.ravel()[-25:])

        elif "UNet" in name:
            model = Unet1D(
                sigmas=torch.tensor([1]),
                dim=64,
                channels= 1,
                dim_mults=(1,1,1),
                use_convnext=True,
                convnext_mult=1,
                with_time_emb=False).to(args.device)
            
            ckpt = torch.load(config["checkpoint"])
            model.load_state_dict(ckpt)
        
        else : raise Exception("Unknown Model Class Name")

        
        models[name] = model

    # 4. Run Evaluation
    print("--- Running Rollouts ---")
    results = evaluate_rollout_with_evaluator(models, m_eval, traj_loader, args.device, args.rollout_steps)
    
    # 5. Compute Metrics & Plotting
    print("--- Computing Metrics ---")
    
    final_metrics = {}
    gt_trajectory = results["data"][:, 1:args.rollout_steps+1] 

    # Setup Figures
    fig_mse, ax_mse = plt.subplots(figsize=(10, 6))
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    fig_eval, ax_eval = plt.subplots(figsize=(10, 6))

    for name in models:
        preds = results["predictions"][name] 
        
        # A. MSE
        mse_time = torch.mean((preds - gt_trajectory)**2, dim=(0,2,3)).cpu().numpy()
        ax_mse.plot(mse_time, label=name)
        
        
        # C. Evaluator Distance
        eval_dist_time = [d.item() for d in results["eval_distances"][name]]
        ax_eval.plot(eval_dist_time, label=name)
        
        # D. Vorticity
        vort_stats = evaluate_trajectory(preds, gt_trajectory)
        ax_corr.plot(vort_stats['mean_correlations'], label=name)
        # Optional: Add error bars/shading
        # ax_corr.fill_between(..., alpha=0.1)
        
        final_metrics[name] = {
            "time_to_failure_avg": vort_stats['time_under_threshold'],
            "time_to_failure_worst10": vort_stats['time_under_threshold_worst_10'],
            "time_to_failure_best10": vort_stats['time_under_threshold_best_10'],
            "final evaluator distance": float(eval_dist_time[-1]),
            "Step 1 MSE": float(mse_time[0]),
            "Step 10 MSE": float(mse_time[10]),
            "Last step MSE": float(mse_time[-1])
        }
        #"final evaluator distance": float(eval_dist_time[-1]),

    # Finalize Plots
    ax_mse.set_yscale('log')
    ax_mse.set_title("MSE vs Time Step")
    ax_mse.set_xlabel("Time Step")
    ax_mse.set_ylabel("MSE")
    ax_mse.legend()
    fig_mse.savefig(os.path.join(args.output_dir, "metric_mse.png"))
    
    ax_eval.set_yscale('log')
    ax_eval.set_title("Distance to Evaluator Model vs Time Step")
    ax_eval.set_xlabel("Time Step")
    ax_eval.legend()
    fig_eval.savefig(os.path.join(args.output_dir, "metric_evaluator_dist.png"))
    
    ax_corr.set_title("Vorticity Correlation vs Time Step")
    ax_corr.set_xlabel("Time Step")
    ax_corr.set_ylabel("Pearson Correlation")
    ax_corr.set_ylim(0, 1.05)
    ax_corr.axhline(0.8, color='black', linestyle='--', alpha=0.5, label='Failure Threshold')
    ax_corr.legend()
    fig_corr.savefig(os.path.join(args.output_dir, "metric_correlation.png"))
    
    # Save Scalar Metrics
    with open(os.path.join(args.output_dir, "metrics_summary.json"), 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()