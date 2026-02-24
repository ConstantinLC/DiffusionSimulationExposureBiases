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
from src.model import Unet
from src.utils import count_parameters, parse_checkpoint_args, run_model, run_model
from src.utils import correlation, vorticity, fsd_torch_radial

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

# Safe correlation function wrapper
def safe_correlation(a, b):
    # Check for NaNs or Infs
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    # Check for zero variance (constant prediction)
    if np.std(a) < 1e-9:
        return 0.0
    return correlation(a, b)
def evaluate_trajectory_vorticity(predictions, ground_truth, threshold=0.8):
    # Safe handling of NaNs
    predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 1. Compute Vorticity (N, T, H, W) -> assumes vorticity() handles batching
    # If your vorticity function expects 4D (B, C, H, W), reshape first
    N, T, C, H, W = predictions.shape
    
    pred_flat = predictions.reshape(-1, C, H, W)
    gt_flat = ground_truth.reshape(-1, C, H, W)

    # Compute vorticity and reshape back to (N, T, H, W)
    # Ensure these are numpy arrays for the math below
    pred_vort = vorticity(pred_flat).reshape(N, T, H, W).cpu().numpy()
    gt_vort = vorticity(gt_flat).reshape(N, T, H, W).cpu().numpy()

    # 2. Vectorized Pearson Correlation
    # Flatten spatial dimensions: (N, T, H*W)
    pred_flat_s = pred_vort.reshape(N, T, -1)
    gt_flat_s = gt_vort.reshape(N, T, -1)

    # Mean centering
    pred_mean = pred_flat_s - pred_flat_s.mean(axis=2, keepdims=True)
    gt_mean = gt_flat_s - gt_flat_s.mean(axis=2, keepdims=True)

    # Covariance (numerator)
    numerator = np.sum(pred_mean * gt_mean, axis=2)

    # Variances (denominator)
    pred_sq_diff = np.sum(pred_mean ** 2, axis=2)
    gt_sq_diff = np.sum(gt_mean ** 2, axis=2)
    denominator = np.sqrt(pred_sq_diff * gt_sq_diff)

    # Avoid division by zero
    epsilon = 1e-8
    corrs = numerator / (denominator + epsilon)

    # Handle flat fields (zero variance) cases
    zero_var_mask = (denominator < epsilon)
    corrs[zero_var_mask] = 0.0

    # 3. Aggregates (Same as before)
    mean_correlations = np.mean(corrs, axis=0)
    std_per_timestep = np.std(corrs, axis=0)

    # Time Until Failure
    below_threshold_mask = corrs < threshold
    
    # Argmax finds the *first* True. If all are False, it returns 0.
    first_failure_indices = np.argmax(below_threshold_mask, axis=1)
    
    # Check if a failure actually occurred
    has_failed = np.any(below_threshold_mask, axis=1)
    
    # If failed, time is index+1. If never failed, time is T.
    times_under_threshold = np.where(has_failed, first_failure_indices + 1, T)

    sorted_times = np.sort(times_under_threshold)
    cutoff_idx = max(1, int(0.1 * N)) 
    worst_10_mean = np.mean(sorted_times[:cutoff_idx])
    top_10_mean = np.mean(sorted_times[-cutoff_idx:])

    return {
        'mean_correlations': mean_correlations,
        'std_per_timestep': std_per_timestep,
        'time_under_threshold': float(np.mean(times_under_threshold)),
        'time_under_threshold_worst_10': float(worst_10_mean),
        'time_under_threshold_best_10': float(top_10_mean),
    }


# ==========================================
# 3. Model Evaluation Logic
# ==========================================

def evaluate_rollout_with_evaluator(models, m_eval, traj_loader, device, rollout_steps=30):
    """
    Runs autoregressive rollout for multiple models over the FULL dataset.
    """
    for model in list(models.values()) + [m_eval]:
        model.eval()

    all_predictions = {name: [] for name in models} 
    all_ground_truth = []
    
    # Track running sum of distances
    running_eval_distances = {name: torch.zeros(rollout_steps, device=device) for name in models}
    total_samples = 0

    print(f"Starting evaluation over full dataset ({len(traj_loader)} batches)...")

    with torch.no_grad():
        for batch_idx, sample in enumerate(traj_loader):
            
            data = sample["data"].to(device) # (B, T_total, C, H, W)
            batch_size = data.shape[0]
            total_samples += batch_size
            
            all_ground_truth.append(data.cpu())

            conditioning_frame = data[:, 0]
            current_preds = {name: run_model(model, conditioning_frame) for name, model in models.items()}
            
            batch_trajectory_buffer = {name: [] for name in models}

            # --- Step 0 ---
            pred_eval_t0 = run_model(m_eval, conditioning_frame)
            
            for name in models:
                batch_trajectory_buffer[name].append(current_preds[name])
                # Safe distance calc
                dist = torch.sum((pred_eval_t0 - current_preds[name]) ** 2, dim=(1,2,3))
                # Clamp locally to prevent running sum exploding to inf
                dist = torch.clamp(dist, max=1e20)
                running_eval_distances[name][0] += dist.sum()

            # --- Autoregressive Rollout ---
            for t in range(1, rollout_steps):
                for name, model in models.items():
                    eval_pred_on_model = run_model(m_eval, current_preds[name])
                    current_preds[name] = run_model(model, current_preds[name])
                    
                    # Store prediction
                    batch_trajectory_buffer[name].append(current_preds[name])
                    
                    # Compute distance
                    diff = (eval_pred_on_model - current_preds[name]) ** 2
                    dist_sum = torch.sum(diff, dim=(1,2,3)).sum()
                    dist_sum = torch.clamp(dist_sum, max=1e20) # Safeguard
                    running_eval_distances[name][t] += dist_sum

            # Store to CPU
            for name in models:
                full_batch_traj = torch.stack(batch_trajectory_buffer[name], dim=1)
                all_predictions[name].append(full_batch_traj.cpu())

            if (batch_idx + 1) % 5 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(traj_loader)}")

    print("Aggregating results...")
    final_predictions = {}
    for name in models:
        final_predictions[name] = torch.cat(all_predictions[name], dim=0)
        
    final_ground_truth = torch.cat(all_ground_truth, dim=0)

    final_eval_distances = {}
    for name in models:
        avg_dist_tensor = running_eval_distances[name] / total_samples
        final_eval_distances[name] = [avg_dist_tensor[t].item() for t in range(rollout_steps)]

    return {
        "predictions": final_predictions,   
        "eval_distances": final_eval_distances, 
        "data": final_ground_truth          
    }


# ==========================================
# 4. Main Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models on Turbulence Data")
    parser.add_argument('--config', type=str, required=True, help="Config file")
    parser.add_argument('--eval_model_path', type=str, default=None, help="Path to the pretrained Evaluator UNet (.pth)")
    parser.add_argument('--checkpoints', nargs='+', required=True, help="List of checkpoints. Format: 'Name=/path/to/ckpt.pth'")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots and metrics")
    parser.add_argument('--rollout_steps', type=int, default=63)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    with open(args.config, 'r') as f: config = json.load(f)
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)

    print(f"--- Evaluating {len(checkpoints_dict)} Models ---")
    for name, path in checkpoints_dict.items():
        print(f"  > {name}: {path}")

    print("--- Loading Data ---")
    _, _, traj_loader = get_data_loaders(config['data_params'])

    print(f"--- Loading Evaluator Model: {args.eval_model_path} ---")
    m_eval = Unet(dim=64, channels=2, dim_mults=(1,1,1), use_convnext=True, convnext_mult=1, with_time_emb=False).to(args.device)
    m_eval.eval()

    models = {}
    for name, ckpt_path in checkpoints_dict.items():
        print(f"--- Loading Candidate: {name} ---")
        if "Diffusion" in name:
            model_config = config['model_params']
            model_config['checkpoint'] = ckpt_path
            model_config['load_betas'] = True
            model = DiffusionModel(**model_config
            ).to(args.device)
        elif "UNet" in name:
            model = Unet(dim=64, channels=2, dim_mults=(1,1,1), use_convnext=True, convnext_mult=1, with_time_emb=False).to(args.device)
            print(f"Loading Checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)
        else: 
            raise Exception("Unknown Model Class Name")
            
        models[name] = model

    print("--- Running Rollouts ---")
    results = evaluate_rollout_with_evaluator(models, m_eval, traj_loader, args.device, args.rollout_steps)
    
    print("--- Computing Metrics ---")
    final_metrics = {}
    gt_trajectory = results["data"][:, 1:args.rollout_steps+1] 

    fig_mse, ax_mse = plt.subplots(figsize=(10, 6))
    fig_fsd, ax_fsd = plt.subplots(figsize=(10, 6))
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    fig_eval, ax_eval = plt.subplots(figsize=(10, 6))

    for name in models:
        preds = results["predictions"][name]
        
        # --- SAFEGUARD: Clamp Predictions ---
        # If the model exploded, values might be +/- Inf or NaN. 
        # We replace NaNs with a large number and clamp Infs.
        preds = torch.nan_to_num(preds, nan=1e5, posinf=1e5, neginf=-1e5)
        preds = torch.clamp(preds, -1e5, 1e5)
        
        # A. MSE
        # Compute squared error
        sq_err = (preds - gt_trajectory)**2
        # Clamp huge errors so mean doesn't become Nan/Inf
        sq_err = torch.clamp(sq_err, max=1e10)
        mse_time = torch.mean(sq_err, dim=(0,2,3,4)).cpu().numpy()
        
        # Ensure numpy array is clean for plotting
        mse_time = np.nan_to_num(mse_time, nan=1e10, posinf=1e10)
        ax_mse.plot(mse_time, label=name)
        
        # B. FSD
        fsd_time = []
        for t in range(args.rollout_steps):
            try:
                val = fsd_torch_radial(preds[:, t], gt_trajectory[:, t])
                val_item = val.item()
                if not np.isfinite(val_item): val_item = 1e5
            except:
                val_item = 1e5
            fsd_time.append(val_item)
        ax_fsd.plot(fsd_time, label=name)
        
        # C. Evaluator Distance
        eval_dist_time = results["eval_distances"][name]
        # Clean up
        eval_dist_time = [min(x, 1e10) if np.isfinite(x) else 1e10 for x in eval_dist_time]
        ax_eval.plot(eval_dist_time, label=name)
        
        # D. Vorticity
        vort_stats = evaluate_trajectory_vorticity(preds, gt_trajectory)
        ax_corr.plot(vort_stats['mean_correlations'], label=name)
        
        # Store metrics (handling potential NaN scalars)
        def clean_scalar(x):
            return float(x) if np.isfinite(x) else 1e10

        final_metrics[name] = {
            "time_to_failure_avg": clean_scalar(vort_stats['time_under_threshold']),
            "time_to_failure_worst10": clean_scalar(vort_stats['time_under_threshold_worst_10']),
            "time_to_failure_best10": clean_scalar(vort_stats['time_under_threshold_best_10']),
            "final FSD": clean_scalar(fsd_time[-1]),
            "final evaluator distance": clean_scalar(eval_dist_time[-1]),
            "Step 1 MSE": clean_scalar(mse_time[0]),
            "Step 10 MSE": clean_scalar(mse_time[10]) if len(mse_time) > 10 else 0.0,
            "Last step MSE": clean_scalar(mse_time[-1])
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