import os
import json
import argparse
import torch
import copy
import wandb
import numpy as np
from torch import nn

# --- Project Imports ---
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.trainer import train_diffusion_model, train_diffusion_model_multisteps
from src.utils import get_next_run_number, count_parameters
from src.diffusion_utils import betas_from_sqrtOneMinusAlphasCumprod

# --- The Schedule Adaptation Function ---
def adapt_schedule(noise_levels, weights, own_pred_errors, prev_pred_errors, clean_errors, tau, incr):
    # Ensure inputs are on CPU for logic processing
    noise_levels = noise_levels.cpu().clone()
    weights = weights.cpu().clone()
    own_pred_errors = own_pred_errors.cpu()
    prev_pred_errors = prev_pred_errors.cpu()
    clean_errors = clean_errors.cpu()

    own_ratio = own_pred_errors / clean_errors
    prev_ratio = prev_pred_errors / clean_errors

    T = len(noise_levels)
    base_weight = 0.1 * T / (T - 1)

    indent = 0
    
    # We iterate and modify locally. 
    # Important: noise_levels must be sorted from clean (low noise) to noisy (high noise) 
    
    for i in range(T):
        if i >= len(own_ratio): break # Safety check

        if own_ratio[i] > tau:
            if i == 0:
                noise_levels[i] = noise_levels[i] * incr
            else:
                weights[i] += base_weight
                weights[0] -= base_weight
        
        elif prev_ratio[i] > tau:
            # Add a new step
            if i < T - 1:
                idx = i + indent
                if idx + 1 >= len(noise_levels): break
                
                new_level = (noise_levels[idx] + noise_levels[idx + 1]) / 2
                
                # Insert level
                noise_levels = torch.cat((noise_levels[:idx+1], torch.tensor([new_level]), noise_levels[idx+1:]))
                
                # Insert weight
                new_weight_tensor = torch.tensor([base_weight])
                weights = torch.cat((weights[:idx+1], new_weight_tensor, weights[idx+1:]))
                
                weights[0] -= base_weight
                indent += 1

    return noise_levels, weights

# --- Evaluation Function ---
def evaluate_model_for_adaptation(model, val_loader, device):
    """
    Computes the error metrics required for adapt_schedule.
    Returns tensors sorted by noise level (Low Noise -> High Noise).
    """
    model.eval()
    mse_ancestor_list = []
    mse_clean_list = []
    mse_own_list = []
    mse_prev_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            data = batch["data"].to(device)
            cond = data[:, 0]
            target = data[:, 1]

            # 1. Ancestor (Ground Truth input)
            _, x0_ancestor = model(conditioning=cond, data=target, return_x0_estimate=True)
            
            # 2. Clean (Ground Truth input)
            _, x0_clean = model(conditioning=cond, data=target, return_x0_estimate=True, input_type="clean")
            
            # 3. Own Pred (Input is model's own output at t)
            _, x0_own = model(conditioning=cond, data=target, return_x0_estimate=True, input_type="own-pred")
            
            # 4. Prev Pred (Input is model's output at t+1)
            _, x0_prev = model(conditioning=cond, data=target, return_x0_estimate=True, input_type="prev-pred")

            # Compute MSE per timestep for this batch
            batch_mse_ancestor = torch.mean((x0_ancestor - target.unsqueeze(0))**2, dim=(1,2,3,4))
            batch_mse_clean = torch.mean((x0_clean - target.unsqueeze(0))**2, dim=(1,2,3,4))
            batch_mse_own = torch.mean((x0_own - target.unsqueeze(0))**2, dim=(1,2,3,4))
            batch_mse_prev = torch.mean((x0_prev - target.unsqueeze(0))**2, dim=(1,2,3,4))
            
            mse_ancestor_list.append(batch_mse_ancestor)
            mse_clean_list.append(batch_mse_clean)
            mse_own_list.append(batch_mse_own)
            mse_prev_list.append(batch_mse_prev)

    # Average over batches
    mse_ancestor = torch.stack(mse_ancestor_list).mean(dim=0).flip(dims=[0])
    mse_clean = torch.stack(mse_clean_list).mean(dim=0).flip(dims=[0])
    mse_own = torch.stack(mse_own_list).mean(dim=0).flip(dims=[0])
    mse_prev = torch.stack(mse_prev_list).mean(dim=0).flip(dims=[0])
    
    return mse_ancestor, mse_own, mse_prev, mse_clean

# --- Main Sequential Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.json')
    parser.add_argument('--rounds', type=int, default=5, help="Number of sequential training rounds")
    parser.add_argument('--tau', type=float, default=1.05, help="Threshold for schedule adaptation")
    parser.add_argument('--log_incr', type=float, default=0.1, help="Logarithmic Increment factor for noise level")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from: {args.config}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initial Setup
    current_config = copy.deepcopy(config)
    
    # Variables to carry over
    current_noise_levels = None 
    current_weights = None

    base_checkpoint_dir = './checkpoints'

    run_number = get_next_run_number(base_checkpoint_dir)

    # 2. Data Loaders
    train_loader, val_loader, traj_loader = get_data_loaders(current_config['data_params'])
    
    for round_idx in range(args.rounds):
        print(f"\n{'='*20} STARTING ROUND {round_idx+1}/{args.rounds} {'='*20}")
        # 1. Setup Directories & Logging
        checkpoint_dir = os.path.join(base_checkpoint_dir, f'run_{run_number}/round_{round_idx+1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Init WandB for this round
        wandb.init(
            project=config['wandb_params']['project'],
            entity=config['wandb_params']['entity'],
            name=f"round_{round_idx+1}_seq",
            group=f"sequential_run_{run_number}",
            config=current_config,
            reinit=True
        )
        
        # 3. Model Initialization
        model = DiffusionModel(**current_config['model_params']).to(device)
        
        # APPLY CUSTOM SCHEDULE (if not first round)
        if round_idx > 0:
            print(f"Applying adapted schedule from previous round...")
            print(f"New Steps: {len(current_noise_levels)}")
            
            # 1. Update timesteps count
            model.timesteps = len(current_noise_levels)
            
            # 2. Update betas/alphas based on noise levels (sqrtOneMinusAlphasCumprod)
            new_betas = betas_from_sqrtOneMinusAlphasCumprod(current_noise_levels.to(device))
            model.compute_schedule_variables(new_betas)
            
            # 3. Update Weights
            model.weights = current_weights.to(device)
            
            # 4. Update U-Net Sigmas (Crucial for preconditioning)
            model.unet.sigmas = (model.sqrtAlphasCumprod / model.sqrtOneMinusAlphasCumprod).ravel()
            
            # 5. Load Weights from Previous Round
            prev_ckpt_path = os.path.join(base_checkpoint_dir, f'run_{run_number}/round_{round_idx}', 'final_model.pth')
            print(f"Loading weights from previous round: {prev_ckpt_path}")
            prev_ckpt = torch.load(prev_ckpt_path)
            # Filter keys to ensure compatibility (removing old sigmas if they exist in state dict)
            prev_ckpt = {key: prev_ckpt[key] for key in prev_ckpt if 'unet' in key and not 'sigmas' in key}
            model.load_state_dict(prev_ckpt, strict=False)
        
        else:
            if config['checkpoint'] != "":
                checkpoint = torch.load(config['checkpoint'])
                checkpoint = {key[5:]:checkpoint[key] for key in checkpoint if 'unet' in key and not 'sigmas' in key}
                model.unet.load_state_dict(checkpoint)
                print(f"Checkpoint loaded from {config['checkpoint']}")

        print(f"Model Parameters: {count_parameters(model)}")
        print("Current noise levels:", model.sqrtOneMinusAlphasCumprod.ravel())
        print("Current weights:", model.weights)
        
        # 4. Train
        criterion = nn.MSELoss()
        
        if current_config['data_params']['sequence_length'][0] == 2:
            train_func = train_diffusion_model
        else:
            train_func = train_diffusion_model_multisteps
            
        trained_model = train_func(
            model, train_loader, val_loader, traj_loader,
            current_config['train_params'], criterion, current_config, checkpoint_dir
        )
        
        # Save Model manually if not handled by trainer
        torch.save(trained_model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))

        # 5. Evaluate & Adapt Schedule
        print("Evaluating for schedule adaptation...")
        mse_ancestor, mse_own, mse_prev, mse_clean = evaluate_model_for_adaptation(trained_model, val_loader, device)

        # --- SAVE SCHEDULE PARAMS ---
        # We extract what was actually used in the trained model for this round
        schedule_save_path = os.path.join(checkpoint_dir, "schedule_params.json")
        schedule_data = {
            "noise_levels": trained_model.sqrtOneMinusAlphasCumprod.ravel().cpu().tolist(),
            "weights": trained_model.weights.cpu().tolist(),
            "timesteps": trained_model.timesteps,
            "mse_ancestor": mse_ancestor.cpu().tolist(),
            "mse_own": mse_own.cpu().tolist(),
            "mse_prev": mse_prev.cpu().tolist(),
            "mse_clean": mse_clean.cpu().tolist()
        }
        with open(schedule_save_path, 'w') as f:
            json.dump(schedule_data, f, indent=4)
        print(f"Saved schedule parameters to {schedule_save_path}")
        # ----------------------------
        
        # Get current noise levels from model (ensure sorted Low -> High)
        current_noise_levels_tensor = trained_model.sqrtOneMinusAlphasCumprod.ravel().cpu()
        current_weights_tensor = trained_model.weights.cpu()
        
        print(f"Old Timesteps: {len(current_noise_levels_tensor)}")
        
        # Adapt
        new_levels, new_weights = adapt_schedule(
            noise_levels=current_noise_levels_tensor,
            weights=current_weights_tensor,
            own_pred_errors=mse_own.cpu(),
            prev_pred_errors=mse_prev.cpu(),
            clean_errors=mse_clean.cpu(),
            tau=args.tau,
            incr=10**args.log_incr
        )
        
        current_noise_levels = new_levels
        current_weights = new_weights
        
        print(f"New Timesteps computed for next round: {len(current_noise_levels)}")
        
        # Update config diffSteps for next logging
        current_config['model_params']['diffSteps'] = len(current_noise_levels)
        
        wandb.finish()
        
        # Free memory
        del model
        del trained_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()