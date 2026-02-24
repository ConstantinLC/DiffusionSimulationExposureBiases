#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Project Imports ---
# Adjust this path if necessary or rely on PYTHONPATH
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.model_1d import Unet1D
from src.utils import run_model

def load_model_from_config(config_path, device):
    """Loads a model based on a configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_params = config['model_params']
    
    # Check if checkpoint path is in config, otherwise warn/error
    if 'checkpoint' not in model_params or model_params['checkpoint'] is None:
        print(f"Warning: No checkpoint found in {config_path}. Model will be random initialized.")
    
    # Determine Model Type (Heuristic based on config content)
    # You might need to adjust this logic based on your specific config structure
    if 'diffusion' in config.get('model_type', 'diffusion').lower():
        model = DiffusionModel(**model_params)
        # For diffusion, we might need to load betas if specified
        if model_params.get('load_betas', False):
            # Assuming compute_schedule_variables logic is handled inside init or here
            pass 
    elif 'unet' in config.get('model_type', '').lower():
         model = Unet1D(**model_params)
         if model_params.get('checkpoint'):
             ckpt = torch.load(model_params['checkpoint'], map_location=device)
             model.load_state_dict(ckpt)
    else:
        # Default fallback to DiffusionModel if unspecified
        model = DiffusionModel(**model_params)

    model.to(device)
    model.eval()
    return model, config

def main():
    parser = argparse.ArgumentParser(description="Visualize Single Channel Prediction from Multiple Configs")
    
    # Arguments
    parser.add_argument('--configs', nargs='+', required=True, 
                        help="List of paths to config JSON files.")
    parser.add_argument('--output_path', type=str, default="channel_comparison.png", 
                        help="Path to save the output image.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample_idx', type=int, default=0, 
                        help="Index of the sample in the batch to visualize.")
    parser.add_argument('--channel_idx', type=int, default=0, 
                        help="Index of the channel to visualize.")
    parser.add_argument('--timesteps', type=int, default=1, 
                        help="Number of autoregressive steps to take (default 1).")
    
    args = parser.parse_args()

    # 1. Load Data
    # We use the parameters from the FIRST config to load the data. 
    # Assumes all models are trained on compatible data.
    print(f"Loading data using config: {args.configs[0]}")
    with open(args.configs[0], 'r') as f:
        first_config = json.load(f)
    
    _, _, traj_loader = get_data_loaders(first_config['data_params'])
    
    # Get one batch
    batch_data = next(iter(traj_loader))
    data = batch_data["data"].to(args.device) # Shape: (B, T_total, C, H)
    
    # Select specific sample
    # Ground Truth: t=0 (Input), t=1..N (Targets)
    conditioning_frame = data[args.sample_idx:args.sample_idx+1, 0] # (1, C, H)
    ground_truth = data[args.sample_idx:args.sample_idx+1, 1:args.timesteps+1] # (1, Steps, C, H)

    # 2. Load Models & Run Inference
    results = {}
    
    for config_path in args.configs:
        model_name = os.path.basename(config_path).replace('.json', '')
        print(f"Processing {model_name}...")
        
        model, _ = load_model_from_config(config_path, args.device)
        
        # Run inference
        current_input = conditioning_frame.clone()
        predictions = []
        
        with torch.no_grad():
            for t in range(args.timesteps):
                # Using the project's run_model utility for abstraction
                pred = run_model(model, current_input)
                predictions.append(pred)
                current_input = pred # Autoregressive step
        
        # Stack predictions: (Steps, C, H) -> remove batch dim since it is 1
        results[model_name] = torch.stack(predictions, dim=1).squeeze(0).cpu().numpy()

    # Move GT to CPU
    conditioning_frame = conditioning_frame.squeeze(0).cpu().numpy()
    ground_truth = ground_truth.squeeze(0).cpu().numpy()

    # 3. Plotting
    num_models = len(results)
    num_cols = 2 + num_models # Input, GT, Model1, Model2, ...
    
    # If multiple timesteps, we plot the LAST timestep
    step_to_plot = args.timesteps - 1
    
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    if num_cols == 1: axes = [axes]
    
    # Global Min/Max for consistent colorbar
    # We aggregate all data to find min/max for the specific channel
    all_data_for_norm = [conditioning_frame[args.channel_idx]]
    all_data_for_norm.append(ground_truth[step_to_plot, args.channel_idx])
    for res in results.values():
        all_data_for_norm.append(res[step_to_plot, args.channel_idx])
        
    vmin = min([x.min() for x in all_data_for_norm])
    vmax = max([x.max() for x in all_data_for_norm])

    # Plot Input
    ax = axes[0]
    im = ax.plot(conditioning_frame[args.channel_idx], label="Input")
    ax.set_title("Input (t=0)")
    ax.grid(True, alpha=0.3)
    
    # Plot Ground Truth
    ax = axes[1]
    ax.plot(ground_truth[step_to_plot, args.channel_idx], label="GT", color='black', linestyle='--')
    ax.set_title(f"Ground Truth (t={step_to_plot+1})")
    ax.grid(True, alpha=0.3)
    
    # Plot Models
    for idx, (name, pred) in enumerate(results.items()):
        ax = axes[idx + 2]
        # Plotting 1D channel
        ax.plot(pred[step_to_plot, args.channel_idx], label=name)
        
        # Optional: Plot GT reference lightly in background
        ax.plot(ground_truth[step_to_plot, args.channel_idx], color='black', linestyle=':', alpha=0.3)
        
        ax.set_title(f"{name}\n(t={step_to_plot+1})")
        ax.set_ylim(vmin, vmax) # Enforce consistent scale
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_path)
    print(f"Saved comparison to {args.output_path}")

if __name__ == "__main__":
    main()