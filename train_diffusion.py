import os
import json
import argparse
import torch
from torch import nn
import wandb
from torch import optim
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.trainer import train_diffusion_model, train_diffusion_model_multisteps
from src.utils import count_parameters
from src.utils import get_next_run_number

def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument('--config', type=str, default='configs/config.json',
                        help='Path to configuration JSON file.')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from: {args.config}")
    
    # --- Setup checkpoint directory ---
    base_checkpoint_dir = './checkpoints'
    run_number = get_next_run_number(base_checkpoint_dir)
    checkpoint_dir = os.path.join(base_checkpoint_dir, f'run_{run_number}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Artifacts for this run will be saved in: {checkpoint_dir}")

    # Save config in checkpoint folder
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # --- Initialize W&B here ---
    project = config['wandb_params']['project']
    entity = config['wandb_params']['entity']
    wandb.init(
        project=project,
        entity=entity,
        name=f"run_{run_number}",
        group=os.path.splitext(os.path.basename(args.config))[0],  # group by config file
        config=config
    )

    # --- Initialize model and data ---
    train_loader, val_loader, traj_loader = get_data_loaders(config['data_params'])
    model = DiffusionModel(**config['model_params'])

    if config['checkpoint'] != "":
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(checkpoint)
        print(f"Checkpoint loaded from {config['checkpoint']}")

    print(f"Model has {count_parameters(model)} parameters.")

    # Initialize loss function
    if config['loss_params']['name'] == 'mse':
        print("Using MSELoss")
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function name: {config['loss_params']['name']}")
    
    # Start training
    if config['data_params']['sequence_length'][0] == 2:
        print('a')
        trained_model = train_diffusion_model(
            model, 
            train_loader, 
            val_loader, 
            traj_loader,
            config['train_params'], 
            criterion, 
            config,
            checkpoint_dir
        )
    else:
        trained_model = train_diffusion_model_multisteps(
            model, 
            train_loader, 
            val_loader, 
            traj_loader,
            config['train_params'], 
            criterion, 
            config,
            checkpoint_dir
        )

    wandb.finish()

if __name__ == '__main__':
    main()
