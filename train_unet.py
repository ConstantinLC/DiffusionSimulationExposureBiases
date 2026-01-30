import os
import json
import argparse
import torch
from torch import nn
import wandb
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Keep your imports
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.trainer import train_unet, train_unet_multisteps
from src.utils import count_parameters
from src.utils import get_next_run_number
from src.model_1d import Unet1D
from src.model import Unet

def setup_ddp():
    # Torchrun sets these environment variables automatically
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank

def cleanup():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument('--config', type=str, default='configs/config.json',
                        help='Path to configuration JSON file.')
    args = parser.parse_args()

    # --- 1. Initialize DDP ---
    local_rank, global_rank = setup_ddp()
    is_master = (global_rank == 0) # Only rank 0 does logging/saving
    device = torch.device("cuda", local_rank)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if is_master:
        print(f"Loaded config from: {args.config}")
    
    # --- Setup checkpoint directory (Only Master) ---
    checkpoint_dir = ""
    if is_master:
        base_checkpoint_dir = config["data_params"]["base_checkpoint_dir"]
        dataset_checkpoint_dir = os.path.join(base_checkpoint_dir, config["data_params"]["dataset_name"])
        run_number = get_next_run_number(dataset_checkpoint_dir)
        checkpoint_dir = os.path.join(dataset_checkpoint_dir, f'run_{run_number}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Artifacts for this run will be saved in: {checkpoint_dir}")

        # Save config
        config_path = os.path.join(checkpoint_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # --- Initialize W&B (Only Master) ---
        project = config['wandb_params']['project']
        entity = config['wandb_params']['entity']
        wandb.init(
            project=project,
            entity=entity,
            name=f"run_{run_number}",
            group=os.path.splitext(os.path.basename(args.config))[0],
            config=config
        )
    
    # Sync processes to ensure directories exist before others proceed (optional but safe)
    dist.barrier()

    # --- Initialize Data ---
    # NOTE: You must update get_data_loaders to accept 'is_distributed=True'
    train_loader, val_loader, traj_loader = get_data_loaders(
        config['data_params'], 
        is_distributed=True # <--- You need to add this logic to your loader
    )

    # --- Initialize Model ---
    model = Unet(**config['model_params'])
    # model = Unet1D(**config['model_params'])

    # Move to specific GPU BEFORE wrapping in DDP
    model.to(device)

    if config['checkpoint'] != "":
        # Map location is important so all GPUs don't load to GPU:0
        checkpoint = torch.load(config['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint)
        if is_master:
            print(f"Checkpoint loaded from {config['checkpoint']}")

    if is_master:
        print(f"Model has {count_parameters(model)} parameters.")

    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank])

    # Initialize loss function
    if config['loss_params']['name'] == 'mse':
        criterion = nn.MSELoss()
    elif config['loss_params']['name'] == 'l1':
        criterion = F.smooth_l1_loss
    else:
        raise ValueError(f"Unknown loss function name: {config['loss_params']['name']}")
    
    # Start training
    # Pass device and rank info to trainer
    if config['data_params']['sequence_length'][0] == 2:
        train_unet(
            model, 
            train_loader, 
            val_loader, 
            traj_loader,
            config['train_params'], 
            criterion, 
            config,
            checkpoint_dir if is_master else None, # Only master gets a path to save
            device=device,
            is_master=is_master
        )

    if config['data_params']['sequence_length'][0] == 3:
        train_unet_multisteps(
            model, 
            train_loader, 
            val_loader, 
            traj_loader,
            config['train_params'], 
            criterion, 
            config,
            checkpoint_dir if is_master else None, # Only master gets a path to save
            device=device,
            is_master=is_master
        )
    
    if is_master:
        wandb.finish()
    
    cleanup()

if __name__ == '__main__':
    main()