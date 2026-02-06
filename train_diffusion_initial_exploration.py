import os
import json
import argparse
import torch
from torch import nn
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Your imports
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.utils import count_parameters, get_next_run_number
from src.trainer_initial_exploration import train_diffusion_single_noise_level

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank

def cleanup():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.json')
    args = parser.parse_args()

    local_rank, global_rank = setup_ddp()
    is_master = (global_rank == 0)
    device = torch.device("cuda", local_rank)

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # --- Setup Directories (Master Only) ---
    if is_master:
        base_dir = config["data_params"]["base_checkpoint_dir"]
        dataset_dir = os.path.join(base_dir, config["data_params"]["dataset_name"])
        run_number = get_next_run_number(dataset_dir)
        run_root_dir = os.path.join(dataset_dir, f'run_{run_number}_binary_search')
        os.makedirs(run_root_dir, exist_ok=True)
        print(f"[Master] Search Artifacts: {run_root_dir}")
    else:
        run_root_dir = "" # Placeholder for workers

    # --- Binary Search Parameters ---
    low_log_sigma = -3.0
    high_log_sigma = -1.0
    epsilon = 0.1  # Stop when the interval is smaller than this
    
    best_passing_sigma = None

    if is_master:
        print(f"--- Starting Binary Search ---")
        print(f"Range: [{low_log_sigma}, {high_log_sigma}] | Target Precision: {epsilon}")

    iteration = 0
    
    while (high_log_sigma - low_log_sigma) > epsilon:
        iteration += 1
        
        # Calculate Midpoint
        mid_log_sigma = (low_log_sigma + high_log_sigma) / 2
        sigma_val_tensor = torch.tensor([10**mid_log_sigma], device=device)
        
        # Cleanup previous step
        dist.barrier()
        
        if is_master:
            print(f"\n=== ITERATION {iteration} ===")
            print(f"Testing Log Sigma: {mid_log_sigma:.4f} (Val: {sigma_val_tensor.item():.5f})")
            print(f"Current Interval: [{low_log_sigma:.4f}, {high_log_sigma:.4f}]")
            
            # Sub-folder for this attempt
            current_checkpoint_dir = os.path.join(run_root_dir, f"iter_{iteration}_sigma_{mid_log_sigma:.3f}")
            os.makedirs(current_checkpoint_dir, exist_ok=True)
            
            wandb.init(
                project=config['wandb_params']['project'],
                entity=config['wandb_params']['entity'],
                name=f"search_{run_number}_iter{iteration}",
                group=f"binary_search_{run_number}",
                config=config,
                reinit=True
            )
        else:
            current_checkpoint_dir = None

        # --- Train Step ---
        # 1. Clean Data Loaders
        train_loader, val_loader, traj_loader = get_data_loaders(config['data_params'], is_distributed=True)

        # 2. Fresh Model
        model = DiffusionModel(**config['model_params'])
        model.to(device)
        
        # 3. Apply Schedule (Single Value)
        model.compute_schedule_variables(sigmas=sigma_val_tensor)
        
        # 4. Wrap DDP
        model = DDP(model, device_ids=[local_rank])
        
        # 5. Criterion
        if config['loss_params']['name'] == 'mse':
            criterion = nn.MSELoss()
        else:
            raise ValueError("Unknown loss")

        # 6. Run Training & Get Boolean Result
        # NOTE: Updates 'train_diffusion_single_noise_level' to return (model, success_bool)
        _, success = train_diffusion_single_noise_level(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            traj_loader=traj_loader,
            train_params=config['train_params'],
            criterion=criterion,
            all_configs=config,
            checkpoint_dir=current_checkpoint_dir,
            device=device,
            is_master=is_master
        )

        if is_master:
            wandb.finish()

        # --- Binary Search Update ---
        # Logic: 
        # If Success (True) -> This sigma works. We want the SMALLEST working one.
        #                      So we try lower. Set high = mid. Record as best so far.
        # If Failure (False)-> This sigma is too small (too hard). We need larger.
        #                      Set low = mid.
        
        if success:
            if is_master: print(f"Result: PASSED. Searching lower interval...")
            best_passing_sigma = mid_log_sigma
            high_log_sigma = mid_log_sigma
        else:
            if is_master: print(f"Result: FAILED. Searching higher interval...")
            low_log_sigma = mid_log_sigma

    # --- Final Result ---
    if is_master:
        print("\n" + "="*40)
        print("BINARY SEARCH COMPLETE")
        print("="*40)
        if best_passing_sigma is not None:
            print(f"Smallest viable Log Sigma: {best_passing_sigma:.5f}")
            print(f"Value: {10**best_passing_sigma:.5f}")
        else:
            print("No viable sigma found in the range (all failed).")
            
    cleanup()

if __name__ == '__main__':
    main()