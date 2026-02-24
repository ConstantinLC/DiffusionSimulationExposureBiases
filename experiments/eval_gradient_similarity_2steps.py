#!/usr/bin/env python
import os
import argparse
import json
import sys
import torch
import torch.nn.functional as F
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import wandb

# --- Project Imports ---
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.utils import count_parameters, parse_checkpoint_args
from src.diffusion_utils import compute_estimate

def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=0)

# ==========================================
# Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Step Gradient Analysis (RGA)")
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to configuration JSON file.')
    parser.add_argument('--checkpoints', nargs='+', required=True, help="List of checkpoints.")
    parser.add_argument('--output_dir', type=str, default="./results_rga_1_4_8", help="Directory to save plots")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Config & Models ---
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    models = {}
    for name, path in checkpoints_dict.items():
        print(f"Loading {name}...")
        model_config = config['model_params']
        model_config['checkpoint'] = path
        model_config['load_betas'] = True
        model = DiffusionModel(**model_config)
        model.to(device)
        model.eval()
        models[name] = model

    print("Loading Data...")
    config['data_params']['val_batch_size'] = 1
    config['data_params']['batch_size'] = 1
    _, _, traj_loader = get_data_loaders(config['data_params'])
    batch_data = next(iter(traj_loader))
    data = batch_data["data"].to(device)
    
    # Needs 9 frames: 0 (Cond) + 1..8 (Targets)
    max_step = 10
    if data.shape[1] < (max_step + 1):
        raise ValueError(f"Dataset needs at least {max_step + 1} frames, got {data.shape[1]}")

    # Ground Truth Trajectory (Indices: 0=Cond, 1=Target1, ..., 8=Target8)
    gt_traj = [data[:, i] for i in range(max_step + 1)]
    
    # Pre-generate noise targets for analysis steps
    noise_targets = {}
    analysis_steps = [2, 5, 10] # The specific steps to analyze
    
    for s in analysis_steps:
        noise_targets[s] = torch.randn_like(gt_traj[s], device=device)

    ddim_steps = list(range(0, 3)) 
    diffusion_steps = 20

    results = {}

    for model_name, model in models.items():
        print(f"\n=== Processing {model_name} ===")
        
        # 1. Baseline Diffusion Rollouts (Full chain 1 -> 8)
        diff_traj = [gt_traj[0]]
        with torch.no_grad():
            curr = gt_traj[0]
            for i in range(1, max_step + 1):
                curr, _ = model(curr, return_x0_estimate=True)
                diff_traj.append(curr)
            
        # 2. Containers for Gradients
        grads = {
            s: {k: {m: [] for m in ["gt", "diff", "est"]} for k in ddim_steps}
            for s in analysis_steps
        }

        # 3. Analysis Loop
        for t in range(diffusion_steps):
            t_tensor = torch.tensor([t], device=device).long()
            alpha = model.sqrtAlphasCumprod[t_tensor]
            sigma = model.sqrtOneMinusAlphasCumprod[t_tensor]
            
            # Pre-compute Noisy Targets
            dNoisy = {}
            for s in analysis_steps:
                dNoisy[s] = alpha * gt_traj[s] + sigma * noise_targets[s]

            for k in ddim_steps:
                # --- A. Generate Estimates Chain ---
                est_traj = [gt_traj[0]]
                with torch.no_grad():
                    curr_est = gt_traj[0]
                    for i in range(1, max_step + 1):
                        est_next = compute_estimate(model, k, curr_est, gt_traj[i])
                        if isinstance(est_next, tuple): est_next = est_next[0]
                        est_traj.append(est_next)
                        curr_est = est_next

                # --- B. Compute Gradients ---
                for s in analysis_steps:
                    ctx_gt   = gt_traj[s-1]
                    ctx_diff = diff_traj[s-1]
                    ctx_est  = est_traj[s-1]
                    
                    target_noise = noise_targets[s]
                    target_noisy_img = dNoisy[s]

                    ctx_map = {"gt": ctx_gt, "diff": ctx_diff, "est": ctx_est}
                    
                    for m, ctx in ctx_map.items():
                        model.unet.zero_grad()
                        # Assuming 2 channels of velocity field. Adjust dimensions if needed.
                        # Input = Context (2ch) + Noisy Target (2ch) -> 4ch
                        inp = torch.cat((ctx, target_noisy_img), dim=1)
                        pred = model.unet(inp, t_tensor)
                        
                        # Assuming model output predicts noise for the target channels
                        # Adjust slicing [-2:] based on your channel setup (e.g. if 2 channels)
                        loss = F.mse_loss(pred[:, -1:], target_noise)
                        loss.backward(retain_graph=True)
                        
                        g = torch.cat([p.grad.flatten().detach() for p in model.unet.parameters() if p.grad is not None])
                        grads[s][k][m].append(g.cpu())

        # 4. Compute RGA Metric
        def compute_rga(step_grads):
            rga_values = []
            for k in ddim_steps:
                means = {m: torch.stack(step_grads[k][m]).mean(dim=0) for m in step_grads[k]}
                
                # Denominator: How well GT aligns with Diff (Baseline)
                sim_gt_diff = cosine_sim(means["gt"], means["diff"])
                
                # Numerator: How well Est aligns with Diff (Candidate)
                sim_est_diff = cosine_sim(means["est"], means["diff"])
                
                # RGA = Candidate / Baseline
                rga = sim_est_diff / (sim_gt_diff + 1e-8)
                rga_values.append(rga.item())
                
            return np.array(rga_values)

        results[model_name] = {s: compute_rga(grads[s]) for s in analysis_steps}

    # ===========================================================
    #  PLOTTING
    # ===========================================================
    print("Generating plots...")
    num_models = len(models)
    
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), squeeze=False, sharey=True)

    # Color Palette for Steps (Blue Gradient)
    # Step 1: Light, Step 4: Medium, Step 8: Dark
    colors = ['#6baed6', '#3182bd', '#08519c'] 
    
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[0, idx]
        x = ddim_steps
        
        # Plot Baseline Reference Line
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='GT Baseline')
        
        # Plot RGA for each analysis step
        for i, step in enumerate(analysis_steps):
            ax.plot(x, res[step], color=colors[i], marker='o', linewidth=2,
                    label=f'Step {step}')

        ax.set_title(f"{name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Lookahead k")
        
        if idx == 0:
            ax.set_ylabel("Relative Gradient Alignment (RGA)\n(>1.0 means Estimate is better than GT)")
            ax.legend(fontsize='medium', frameon=True)
            
        ax.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "rga_analysis_1_4_8.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()