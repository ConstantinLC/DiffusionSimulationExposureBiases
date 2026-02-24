import torch
import torch.optim as optim
import wandb
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..utils.general import traj_eval_step

def train_unet(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir, device, is_master):
    """
    Trains a U-Net model with DDP support.
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    
    # --- Config extraction ---
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    # Note: Model is already moved to device in main.py before DDP wrapping
    # But strictly ensuring input data goes to the right device is handled in the loop.

    # --- 1. Optimizer & Scheduler Setup ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    
    # T_max is the number of epochs until the LR reaches the minimum
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_params["T_max"], 
        eta_min=lr_end
    )

    if is_master:
        print(f"Starting Training on {device} for {num_epochs} epochs...")
        print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")

    best_traj_error = float('inf')
    best_traj_time = 0

    for epoch in range(num_epochs):
        if epoch >= train_params["T_max"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_end

        # --- DDP: Set Epoch for Sampler ---
        # Crucial for shuffling to work correctly across GPUs
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]
            
            optimizer.zero_grad()
            pred = model(conditioning_frame, time=None)
            loss = criterion(pred, target_frame)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # --- Validation Loop ---
        # Note: In DDP, each GPU calculates val loss on its subset. 
        # We typically just log Rank 0's validation loss to save overhead.
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            # If using DistributedSampler for validation, set epoch as well
            if hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)
                
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data_val = sample_val["data"].to(device)
                    if data_val.shape[1] != 2:
                        continue
                    conditioning_frame_val = data_val[:, 0]
                    target_frame_val = data_val[:, 1]
                    
                    pred = model(conditioning_frame_val, time=None)
                    loss_val = criterion(pred, target_frame_val)
                    running_val_loss += loss_val.item()
            
            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- 2. Step Scheduler & Get Current LR ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Logging (Master Only) ---

        model_to_eval = model.module if isinstance(model, DDP) else model

        log_dict = {
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "learning_rate": current_lr,
            "best_traj_time": best_traj_time,
            "best_traj_error": best_traj_error
        }
            
        # Run trajectory evaluation periodically (only on master)
        log_dict = traj_eval_step(
            traj_loader, 
            epoch, 
            epoch_sampling_frequency, 
            model_to_eval, 
            device, 
            all_configs, 
            log_dict, 
            checkpoint_dir
        )

        # --- Logging (Master Only) ---
        if is_master:
            # Update local bests (traj_eval_step updates the log_dict)
            if 'best_traj_time' in log_dict:
                best_traj_time = log_dict['best_traj_time']
            if 'best_traj_error' in log_dict:
                best_traj_error = log_dict['best_traj_error']

            wandb.log(log_dict)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

        if isinstance(model, DDP):
            import torch.distributed as dist
            dist.barrier()

    if is_master:
        print("Training complete!")
        
    return model



def train_unet_multisteps(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir, device, is_master):
    """
    Trains a U-Net model using unrolled (autoregressive) training steps.
    The model predicts the next frame, and that prediction is fed back 
    as input for the subsequent step calculation.
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    
    # --- Config extraction ---
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    # --- 1. Optimizer & Scheduler Setup ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_params["T_max"], 
        eta_min=lr_end
    )

    if is_master:
        print(f"Starting Multi-Step U-Net Training on {device} for {num_epochs} epochs...")
        print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")

    best_traj_error = float('inf')
    best_traj_time = 0

    for epoch in range(num_epochs):
        # Enforce minimum LR after T_max
        if epoch >= train_params["T_max"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_end

        # --- DDP: Set Epoch for Sampler ---
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device) # Shape: (B, T, C, H, W)
            optimizer.zero_grad()
            
            # Accumulate loss over the trajectory
            loss = torch.tensor(0.0, device=device)
            
            # Initial input is the first ground truth frame
            current_input = data[:, 0]

            # Iterate through time steps (0 -> 1, 1 -> 2, ...)
            # We predict T-1 transitions
            steps = data.shape[1] - 1
            
            for t in range(steps):
                target_frame = data[:, t + 1]

                # Forward pass: Predict next frame
                # U-Net typically ignores the 'time' argument used in diffusion
                pred = model(current_input, time=None)
                
                # Accumulate MSE loss
                loss += criterion(pred, target_frame)
                
                # Autoregressive step: Use prediction as input for next step
                # (Detach to prevent gradient explosion if sequence is very long, 
                # though usually for short unrolls we keep gradients)
                current_input = pred

            # Backpropagation on the accumulated loss
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # --- Validation Loop ---
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            if hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)
                
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data_val = sample_val["data"].to(device)
                    if data_val.shape[1] < 2:
                        continue
                        
                    loss_val = torch.tensor(0.0, device=device)
                    current_input_val = data_val[:, 0]
                    steps_val = data_val.shape[1] - 1

                    for t in range(steps_val):
                        target_frame_val = data_val[:, t + 1]
                        pred_val = model(current_input_val, time=None)
                        loss_val += criterion(pred_val, target_frame_val)
                        current_input_val = pred_val

                    running_val_loss += loss_val.item()
            
            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- Scheduler Step ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Logging (Master Only) ---
        if is_master:
            log_dict = {
                "epoch": epoch + 1,
                "training_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "learning_rate": current_lr,
                "best_traj_time": best_traj_time,
                "best_traj_error": best_traj_error
            }
            
            # Access underlying model if wrapped in DDP
            if isinstance(model, DDP):
                model_to_eval = model.module
            else:
                model_to_eval = model

            # Run trajectory evaluation
            log_dict = traj_eval_step(
                traj_loader, 
                epoch, 
                epoch_sampling_frequency, 
                model_to_eval, 
                device, 
                all_configs, 
                log_dict, 
                checkpoint_dir
            )

            # Update local bests
            if 'best_traj_time' in log_dict:
                best_traj_time = log_dict['best_traj_time']
            if 'best_traj_error' in log_dict:
                best_traj_error = log_dict['best_traj_error']

            wandb.log(log_dict)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

    if is_master:
        print("Training complete!")
        
    return model