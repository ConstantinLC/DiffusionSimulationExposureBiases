import os
import json
import argparse
import torch
from torch import nn
import wandb
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
# Import the specific training function you asked for
from src.trainer_exploration import obtain_level_lines_tau
from src.utils import count_parameters, get_next_run_number

def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model with schedule pruning.")
    parser.add_argument('--config', type=str, default='configs/config.json',
                        help='Path to configuration JSON file.')
    args = parser.parse_args()

    # --- 1. Load Config ---
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from: {args.config}")
    
    # --- 2. Setup Checkpoint Directory ---
    base_checkpoint_dir = './checkpoints'
    run_number = get_next_run_number(base_checkpoint_dir)
    checkpoint_dir = os.path.join(base_checkpoint_dir, f'run_{run_number}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Artifacts for this run will be saved in: {checkpoint_dir}")

    # Save the config used for this run
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # --- 3. Initialize W&B ---

    # --- 4. Initialize Data & Model ---
    train_loader, val_loader, traj_loader = get_data_loaders(config['data_params'])
    model = DiffusionModel(**config['model_params'])

    print(f"Model has {count_parameters(model)} parameters.")

    # --- 5. Checkpoint Loading (Optional) ---
    if config.get('checkpoint', ""):
        ckpt_path = config['checkpoint']
        print(f"Loading checkpoint from {ckpt_path}...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=config['train_params']['device'])
            
            # Helper to clean keys if needed (handling 'unet.' prefix vs direct saves)
            clean_state_dict = {}
            for key, val in checkpoint.items():
                # If key starts with 'unet.', strip it to match model.unet
                if key.startswith('unet.'):
                    new_key = key.replace("unet.", "")
                    clean_state_dict[new_key] = val
                # Or if it's a direct save (no 'unet' in key name usually, but avoid 'sigmas')
                elif 'sigmas' not in key:
                    clean_state_dict[key] = val
            
            # Load into the UNet specifically
            model.unet.load_state_dict(clean_state_dict, strict=False)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")

    # --- 6. Loss Function ---
    if config['loss_params']['name'] == 'mse':
        print("Using MSELoss")
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function name: {config['loss_params']['name']}")
    
    # --- 7. Validate Config for New Trainer ---
    # Ensure start/end LRs exist, or fallback to the old 'learning_rate'
    if "learning_rate_start" not in config['train_params']:
        lr = config['train_params'].get('learning_rate', 1e-4)
        print(f"Warning: 'learning_rate_start' missing. Using 'learning_rate' ({lr}) as start.")
        config['train_params']['learning_rate_start'] = lr

    if "learning_rate_end" not in config['train_params']:
        lr_end = config['train_params'].get('learning_rate', 1e-4) / 100
        print(f"Warning: 'learning_rate_end' missing. Using {lr_end} as end.")
        config['train_params']['learning_rate_end'] = lr_end

    # --- 8. Start Training ---
    print("Starting Initial Exploration Training (Schedule Pruning)...")
    
    trained_model = obtain_level_lines_tau(
        model, 
        train_loader, 
        val_loader, 
        config['train_params'], 
        criterion, 
        config,
        checkpoint_dir
    )

if __name__ == '__main__':
    main()