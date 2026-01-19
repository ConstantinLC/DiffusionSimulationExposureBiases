import torch
import torch.optim as optim
import wandb
import os
from src.utils import evaluate_trajectory_vorticity, evaluate_trajectory_mse
from src.diffusion_utils import compute_estimate, betas_from_sqrtOneMinusAlphasCumprod
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
import json
import collections


"""def get_error_bucket(ratio):
    if 1.08 <= ratio < 1.10:
        return "1.08-1.10"
    elif 1.06 <= ratio < 1.08:
        return "1.06-1.08"
    elif 1.04 <= ratio < 1.06:
        return "1.04-1.06"
    elif 1.02 <= ratio < 1.04:
        return "1.02-1.04"
    elif ratio < 1.02:
        return "<1.02"
    return None"""

def get_error_bucket(ratio):
    """Classifies the error ratio into specific ranges."""
    if 1.005 <= ratio < 1.01:
        return "1.01"
    if 1.025 <= ratio < 1.03:
        return "1.03"
    elif 1.055 <= ratio < 1.06:
        return "1.05"
    elif 1.105 <= ratio < 1.11:
        return "1.1"
    elif 1.155 <= ratio < 1.16:
        return "1.15"
    return None




def train_diffusion_model_initial_exploration_onebyone(model, train_loader, val_loader, train_params, criterion, all_configs, checkpoint_dir):
    """
    Trains a diffusion model with Cosine Annealing and Dynamic Schedule Pruning.
    Optimized to save a single checkpoint per epoch for multiple pruned steps.
    """
    device = torch.device(train_params["device"])
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    # Pruning threshold
    tau = 1.05
    
    model.to(device)

    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    t_max = train_params.get("T_max", num_epochs)
    #scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr_end)

    scheduler = StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)

    print(f"Starting training on {device} for {num_epochs} epochs.")
    print(f"Schedule: Cosine {lr_start:.1e} -> {lr_end:.1e} (Tau={tau})")

    # Storage for results
    eb_free_error = {}  # Maps alpha -> clean error
    eb_free_checkpoints = {} # Maps alpha -> checkpoint path

    current_timestep = 10 #model.timesteps-1

    for epoch in range(num_epochs):
        # ==========================
        # 1. Training Loop
        # ==========================
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]
            
            optimizer.zero_grad()
            noise, predicted_noise = model(conditioning_frame, target_frame, fixed_timestep=current_timestep)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # ==========================
        # 2. Validation & Pruning
        # ==========================
        avg_val_loss = 0.0
        
        if val_loader is not None:
            # --- B. Pruning Check (Single Batch) ---
            check_batch = next(iter(val_loader))
            data_check = check_batch["data"].to(device)
            cond_check = data_check[:, 0]
            target_check = data_check[:, 1]
            
            with torch.no_grad():
                own_pred = model(cond_check, target_check, 
                                                  fixed_timestep=current_timestep, 
                                                  return_x0_estimate=True,
                                                  input_type="input-own-pred")
                    
                clean_pred = model(cond_check, target_check, 
                                                fixed_timestep=current_timestep, 
                                                return_x0_estimate=True,
                                                input_type="clean")
                
                clean_error = criterion(clean_pred, target_check).item()
                own_pred_error = criterion(own_pred, target_check).item()
                
                ratio = own_pred_error / clean_error

                print(model.sqrtOneMinusAlphasCumprod.ravel()[current_timestep].item(), ratio, clean_error)
                
                if ratio < tau:
                    nl_val = model.sqrtOneMinusAlphasCumprod.ravel()[current_timestep].item()
                    current_timestep -= 1
                    print(nl_val, ratio, clean_error, own_pred_error)
                    # Store error immediately
                    eb_free_error[nl_val] = clean_error

                    # 1. Save ONE checkpoint for all steps pruned in this epoch
                    ckpt_name = f"pruned_epoch_{epoch}.pt"
                    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                    torch.save(model.unet.state_dict(), ckpt_path)
                    
                    eb_free_checkpoints[nl_val] = ckpt_path

                    print(f"{nl_val} completed at Epoch {epoch+1}. Saved to {ckpt_name}")

                    map_path = os.path.join(checkpoint_dir, 'checkpoint_map.json')
                    with open(map_path, 'w') as f:
                        json.dump(collections.OrderedDict(sorted(eb_free_checkpoints.items())), f, indent=4)
                    print(f"Checkpoint map saved to {map_path}")

                    map_path = os.path.join(checkpoint_dir, 'error_map.json')
                    with open(map_path, 'w') as f:
                        json.dump(collections.OrderedDict(sorted(eb_free_error.items())), f, indent=4)
                    print(f"Checkpoint map saved to {map_path}")

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.0001

                    # 2. THEN create the new scheduler
                    # Now it sees 0.001 as the base_lr and starts fresh
                    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

        # ==========================
        # 3. Scheduler & Logging
        # ==========================
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | "
              f"LR: {current_lr:.2e} | Steps Left: {model.timesteps}")

    print("Training complete!")
    return model



def obtain_level_lines_tau(model, train_loader, val_loader, train_params, criterion, all_configs, checkpoint_dir):
    """
    Trains a diffusion model with strict pruning < 1.02.
    - Tracks error milestones in specific buckets ([1.08-1.10], etc) without overwriting.
    - ONLY deletes steps when error ratio < 1.02.
    - DOES NOT save intermediate checkpoints during the pruning loops.
    """
    device = torch.device(train_params["device"])
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    model.to(device)

    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    scheduler = CosineAnnealingLR(optimizer, T_max=101, eta_min=lr_end)

    print(f"Starting training on {device} for {num_epochs} epochs.")

    eb_free_error_tracking = {} 

    for epoch in range(num_epochs):
        # ==========================
        # 1. Training Loop
        # ==========================
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]
            
            optimizer.zero_grad()
            # Standard training step
            noise, predicted_noise = model(conditioning_frame, target_frame)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # ==========================
        # 2. Validation & Pruning Logic
        # ==========================
        avg_val_loss = 0.0
        steps_to_delete = []
        
        if val_loader is not None:
            # --- A. Standard Validation Loss ---

            # --- B. Error Ratio Check (Single Representative Batch) ---
            check_batch = next(iter(val_loader))
            data_check = check_batch["data"].to(device)
            cond_check = data_check[:, 0]
            target_check = data_check[:, 1]
            
            with torch.no_grad():
                # Iterate over currently active timesteps
                current_timesteps = range(model.timesteps)
                
                for ts in current_timesteps:
                    print(ts)
                    # 1. Get noise level (sigma/alpha) for this step
                    nl_val = model.sqrtOneMinusAlphasCumprod.ravel()[ts].item()
                    
                    # 2. Compute Ratios
                    own_pred = model(cond_check, target_check, 
                                     fixed_timestep=ts, 
                                     return_x0_estimate=True, 
                                     input_type="input-own-pred")
                    
                    clean_pred = model(cond_check, target_check, 
                                       fixed_timestep=ts, 
                                       return_x0_estimate=True, 
                                       input_type="clean")
                    
                    clean_error = criterion(clean_pred, target_check).item()
                    own_pred_error = criterion(own_pred, target_check).item()
                    
                    # Prevent division by zero
                    if clean_error < 1e-9:
                        ratio = 1.0
                    else:
                        ratio = own_pred_error / clean_error

                    # 3. Bucket Logic (Store only if new for this bucket)
                    bucket = get_error_bucket(ratio)
                    
                    if bucket:
                        # Initialize dict key if missing
                        if nl_val not in eb_free_error_tracking:
                            eb_free_error_tracking[nl_val] = {}
                        
                        # Only write if this bucket is empty for this noise level
                        if bucket not in eb_free_error_tracking[nl_val]:
                            eb_free_error_tracking[nl_val][bucket] = {
                                "ratio": ratio,
                                "clean_error": clean_error,
                                "own_error": own_pred_error,
                                "epoch": epoch
                            }
                            # Optional debug print
                            # print(f"  [Log] Step {ts} (NL {nl_val:.4f}): Captured bucket {bucket} (Ratio: {ratio:.4f})")

            # --- C. Save Error Logs (JSON only) ---
            map_path = os.path.join(checkpoint_dir, 'error_tracking_map.json')
            with open(map_path, 'w') as f:
                # Sort keys for readability
                json.dump(collections.OrderedDict(sorted(eb_free_error_tracking.items())), f, indent=4)

        # ==========================
        # 3. Scheduler & Logging
        # ==========================
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"LR: {current_lr:.2e} | Steps Left: {model.timesteps}")

    print("Training complete!")
    return model