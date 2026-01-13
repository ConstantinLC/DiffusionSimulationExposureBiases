import os
import json
import argparse
import torch
from torch import nn
import wandb
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
# Import the specific training function you asked for
from src.trainer import train_diffusion_model_initial_exploration
from src.utils import count_parameters, get_next_run_number
from src.diffusion_utils import run_dynamic_checkpoint_inference

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
    wandb.init(
        project=config['wandb_params']['project'],
        entity=config['wandb_params']['entity'],
        name=f"run_{run_number}",
        group="schedule_exploration",  # Useful grouping for these specific runs
        config=config
    )

    # --- 4. Initialize Data & Model ---
    train_loader, val_loader, traj_loader = get_data_loaders(config['data_params'])
    model = DiffusionModel(**config['model_params'])

    print(f"Model has {count_parameters(model)} parameters.")

    # 1. Load the map generated during training
    # If you saved it to JSON:
    with open("checkpoints/run_474/checkpoint_map.json", "r") as f:
        # Keys in JSON are always strings, convert back to float
        loaded_map = json.load(f)
        checkpoint_map = {float(k): v for k, v in loaded_map.items()}

    # 2. Prepare input (e.g., from validation loader)
    sample = next(iter(val_loader))
    conditioning = sample["data"][:, 0]  # Shape: [B, C, H, W]
    target = sample["data"][:, 1].to('cuda')

    # 3. Run Inference
    final_prediction, clean_prediction = run_dynamic_checkpoint_inference(
        model=model,
        conditioning_frame=conditioning,
        target_frame=target,
        checkpoint_map=checkpoint_map,
        device="cuda"
    )

    # 4. Visualize or calculate metrics
    print("Inference Complete. Output shape:", final_prediction.shape)

    # --- 6. Loss Function ---
    if config['loss_params']['name'] == 'mse':
        print("Using MSELoss")
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function name: {config['loss_params']['name']}")
    
    print("Prediction error:", criterion(final_prediction, target))
    print("Clean Prediction error:", criterion(clean_prediction, target))

    wandb.finish()

if __name__ == '__main__':
    main()

