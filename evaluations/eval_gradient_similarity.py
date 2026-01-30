#!/usr/bin/env python
import os
import argparse
import json
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import wandb

# --- Project Imports ---
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.utils import count_parameters, parse_checkpoint_args
from src.diffusion_utils import compute_estimate

# Set plotting style
sns.set_theme(style="whitegrid")

def cosine_sim(a, b):
    # Returns scalar cosine similarity
    return F.cosine_similarity(a, b, dim=0).item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mean Gradient Similarity")
    parser.add_argument('--config', type=str, default='configs/config.json',
                        help='Path to configuration JSON file.')
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                        help="List of checkpoints. Format: 'Name=/path/to/ckpt.pth'. Space separated.")
    parser.add_argument('--output_dir', type=str, default="./results_gradients_baseline_check", help="Directory to save plots")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Checkpoints ---
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)
    
    # --- Load Config ---
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # --- Initialize Models ---
    models = {}
    for name in checkpoints_dict:
        print(f"Loading {name}...")
        model_config = config['model_params']
        model_config['checkpoint'] = checkpoints_dict[name]
        model_config['load_betas'] = True

        model = DiffusionModel(**model_config)
        model.to(device)
        model.eval()
        models[name] = model

    # --- Load Data ---
    print("Loading Data...")
    _, _, traj_loader = get_data_loaders(config['data_params'])
    batch_data = next(iter(traj_loader))
    data = batch_data["data"].to(device)
    
    conditioning_frame = data[:, 0]
    target_frame = data[:, 1]
    gt_2nd_step = data[:, 2]

    # Evaluation Params
    diffusion_steps = 20
    ddim_max_steps = 0
    # CHANGED: Increased range to 10 steps
    ddim_steps = list(range(ddim_max_steps, ddim_max_steps + 10)) 
    
    num_samples = 5 
    
    results = {}
    model_sigmas = {}

    # Pre-generate noise for the reconstruction loss target
    noise_base = torch.randn_like(gt_2nd_step, device=device)

    for model_name, model in models.items():
        print(f"\n=== Processing model: {model_name} ===")

        sigmas = model.sqrtOneMinusAlphasCumprod.ravel()
        sigmas = torch.concatenate((torch.tensor([sigmas[0]]).to('cuda'), sigmas[-19:]))
        model.compute_schedule_variables(sigmas = sigmas)
        
        alphas = model.sqrtOneMinusAlphasCumprod.ravel().cpu().numpy()
        model_sigmas[model_name] = np.min(alphas)
        
        # Lists to store results across timesteps
        # k -> list of scores
        scores_est_vs_diffA = {k: [] for k in ddim_steps}
        scores_diffA_vs_diffB = [] 

        for t in range(diffusion_steps):
            t_tensor = torch.tensor([t], device=device).long()
            
            # Create Noisy Input at step t
            alpha = model.sqrtAlphasCumprod[t_tensor]
            sigma = model.sqrtOneMinusAlphasCumprod[t_tensor]
            dNoisy = alpha * gt_2nd_step + sigma * noise_base

            # --- A. Compute Mean Gradient for Set A (5 Samples) ---
            mean_grad_A = None
            for i in range(num_samples):
                with torch.no_grad():
                    sample, _ = model(conditioning_frame, return_x0_estimate=True)
                
                model_input = torch.cat((sample, dNoisy), dim=1)
                model.unet.zero_grad()
                pred_noise = model.unet(model_input, t_tensor)
                loss = F.smooth_l1_loss(pred_noise[:, -5:], noise_base)
                loss.backward()
                
                grad = torch.cat([p.grad.flatten().detach() for p in model.unet.parameters() if p.grad is not None])
                
                if mean_grad_A is None: mean_grad_A = grad / num_samples
                else: mean_grad_A += grad / num_samples

            # --- D. Compute Mean Gradient for Proxy Estimates (per k) ---
            for k in ddim_steps:
                mean_est_grad = None
                
                for i in range(num_samples):
                    with torch.no_grad():
                        est = compute_estimate(model, k, conditioning_frame, target_frame)
                        if isinstance(est, tuple): est = est[0]

                    model_input = torch.cat((est, dNoisy), dim=1)
                    model.unet.zero_grad()
                    pred_noise = model.unet(model_input, t_tensor)
                    loss = F.smooth_l1_loss(pred_noise[:, -5:], noise_base)
                    loss.backward()
                    
                    grad = torch.cat([p.grad.flatten().detach() for p in model.unet.parameters() if p.grad is not None])
                    
                    if mean_est_grad is None: mean_est_grad = grad / num_samples
                    else: mean_est_grad += grad / num_samples

                # Compare Estimate Mean to Diffusion Mean (Set A)
                sim_est = cosine_sim(mean_grad_A, mean_est_grad)
                scores_est_vs_diffA[k].append(sim_est)

        avg_est_scores = []
        for k in ddim_steps:
            avg_est_scores.append(np.mean(scores_est_vs_diffA[k]))
            
        results[model_name] = {
            "estimates": np.array(avg_est_scores)
        }

    # ===========================================================
    #  PLOTTING
    # ===========================================================
    print("Generating combined plot...")
    
    # CHANGED: Increased width for more horizontal aspect ratio
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    colors = sns.color_palette("deep", max(10, len(results)))
    
    # Larger font sizes
    FONT_SIZE_AXIS = 18
    FONT_SIZE_TICKS = 16
    FONT_SIZE_LEGEND = 16
    FONT_SIZE_ANNOT = 16

    for idx, (model_name, res) in enumerate(results.items()):
        color_main = colors[idx % len(colors)]
        sigma_min = model_sigmas[model_name]
        
        # Plot Curve
        ax.plot(ddim_steps, res["estimates"], label=model_name, 
                linestyle='-', linewidth=3.0, color=color_main, marker='o', markersize=9)

        # Annotation: Sigma min
        # Stacked vertically on the right side
        vertical_offset = 0.05 + (idx * 0.08)
        ax.text(0.98, vertical_offset, f"$\sigma_{{min}}={sigma_min:.1e}$", 
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=FONT_SIZE_ANNOT, fontweight='bold', color=color_main,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Styling
    ax.set_ylabel("Cosine Similarity", fontsize=FONT_SIZE_AXIS)
    ax.set_xlabel("Proxy Estimate Steps (n)", fontsize=FONT_SIZE_AXIS)
    ax.set_xticks(ddim_steps)
    
    # Tick sizing
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    # CHANGED: Legend at the bottom, outside the plot
    # bbox_to_anchor coordinates are (x, y) relative to the axes. y < 0 moves it below.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=len(results), fontsize=FONT_SIZE_LEGEND, frameon=True)

    # Adjust layout to accommodate the external legend
    plt.subplots_adjust(bottom=0.2)
    
    # Save
    save_path = os.path.join(args.output_dir, "gradient_similarity_combined.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    
    print(f"Plot saved to {save_path}")

    if wandb.run is not None:
        wandb.log({"Gradient_Analysis_Combined": wandb.Image(save_path)})
        wandb.finish()

if __name__ == '__main__':
    main()