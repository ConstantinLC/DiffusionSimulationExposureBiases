import torch
import torch.optim as optim
import wandb
import os
from src.utils import evaluate_trajectory_vorticity, evaluate_trajectory_mse
from src.diffusion_utils import compute_estimate
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import collections

def traj_eval_step(traj_loader, epoch, epoch_sampling_frequency, model, device, all_configs, log_dict, checkpoint_dir):

    # --- Trajectory Evaluation  and Logging ---
    if traj_loader and epoch % epoch_sampling_frequency == 0 and epoch > 0:
        model.eval()
        print("Evaluating on trajectories...")

        if all_configs["loss_params"]["eval_traj_metric"] == "vorticity_corr":
            traj_metrics = evaluate_trajectory_vorticity(model, traj_loader, device)
            current_traj_time = traj_metrics['time_under_threshold']
            
            # Log the current trajectory metrics
            log_dict['time_under_0.8'] = current_traj_time
            for t, corr in enumerate(traj_metrics['mean_correlations']):
                log_dict[f'vorticity_correlation_t_{t+1}'] = corr
            print(f"🌀 Vorticity correlation time under 0.8: {current_traj_time}")

            if current_traj_time > log_dict['best_traj_time']:
                best_traj_time = current_traj_time
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ New best model saved to {best_checkpoint_path} with trajectory time: {best_traj_time}")

                log_dict['best_traj_time'] = best_traj_time

        elif all_configs["loss_params"]["eval_traj_metric"] == "mse":
            traj_metrics = evaluate_trajectory_mse(model, traj_loader, device)
            current_traj_error = traj_metrics['mean_error']

            for t, mse in enumerate(traj_metrics['errors_per_ts']):
                log_dict[f'mse_t_{t+1}'] = mse

            if current_traj_error < log_dict['best_traj_error']:
                best_traj_error = current_traj_error
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ New best model saved to {best_checkpoint_path} with trajectory error: {best_traj_error}")

                log_dict['best_traj_error'] = best_traj_error

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved for epoch {epoch+1} at {checkpoint_path}")

    return log_dict


def train_diffusion_model(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir):
    """
    Trains a diffusion model with Cosine Annealing Learning Rate.
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    device = torch.device(train_params["device"])
    
    # --- Config extraction ---
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    model.to(device)

    # --- 1. Optimizer & Scheduler Setup ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    
    # T_max is the number of epochs until the LR reaches the minimum
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_params["T_max"], 
        eta_min=lr_end
    )

    print(f"Starting Diffusion training on {device} for {num_epochs} epochs...")
    print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")

    best_traj_error = float('inf')
    best_traj_time = 0

    for epoch in range(num_epochs):
        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]
            
            optimizer.zero_grad()
            noise, predicted_noise = model(conditioning_frame, target_frame)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # --- Validation Loop ---
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data_val = sample_val["data"].to(device)
                    if data_val.shape[1] != 2:
                        continue
                    conditioning_frame_val = data_val[:, 0]
                    target_frame_val = data_val[:, 1]
                    
                    noise, predicted_noise = model(conditioning_frame_val, target_frame_val)
                    loss_val = criterion(predicted_noise, noise)
                    running_val_loss += loss_val.item()
            
            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- 2. Step Scheduler & Get Current LR ---
        # Get the LR before stepping for logging accuracy
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Logging ---
        log_dict = {
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "learning_rate": current_lr,  # Log LR to visualize the cosine curve
            "best_traj_time": best_traj_time,
            "best_traj_error": best_traj_error
        }
        
        # Run trajectory evaluation periodically
        log_dict = traj_eval_step(traj_loader, epoch, epoch_sampling_frequency, model, device, all_configs, log_dict, checkpoint_dir)

        wandb.log(log_dict)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.2e}")

    print("Training complete!")
    return model



def train_diffusion_model_multisteps(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir):
    """
    Trains a diffusion model
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    best_val_loss = float('inf')
    best_traj_time = 0
    best_traj_error = 100000000
    optimizer = optim.Adam(model.parameters(), lr=train_params["learning_rate"])
    device = torch.device(train_params["device"])
    model.to(device)
    n_proxy_steps = train_params["n_proxy_steps"]
    backgrad = train_params["backgrad"]

    print(f"Starting Multi-steps Diffusion training on {device} for {train_params['num_epochs']} epochs...")

    for epoch in range(train_params["num_epochs"]):
        # (Training and validation loops remain unchanged)
        model.train()
        running_train_loss = 0.0
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device)
            optimizer.zero_grad()
            loss = torch.Tensor([0]).to(device)

            for t in range(data.shape[1] - 1):
                if t == 0:
                    conditioning_frame = data[:, t]
                else:
                    conditioning_frame = x0_estimate
                #conditioning_frame = data[:, t]
                target_frame = data[:, t + 1]
            
                if t < data.shape[1] - 2:
                    noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)

                    # Obtain EOS estimate for next AR step  
                   # with torch.no_grad():
                   #     _, _, x0_estimate = model(conditioning_frame, target_frame, return_x0_estimate=True, limit_timesteps_training=train_params["first_ar_step_noising_step_limit"])
                    if backgrad:
                        x0_estimate = compute_estimate(model, n_proxy_steps, conditioning_frame, target_frame)
                    else:
                        with torch.no_grad():
                            
                            x0_estimate = compute_estimate(model, n_proxy_steps, conditioning_frame, target_frame)

                if t > 0:
                    noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)

                loss += criterion(predicted_noise, noise)

            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = running_train_loss / (batch_idx + 1)
        print(avg_train_loss)
        
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data = sample["data"].to(device)
                    loss = torch.Tensor([0]).to(device)

                    for t in range(data.shape[1] - 1):
                        """if t == 0:
                            conditioning_frame = data[:, t]
                        else:
                            conditioning_frame = x0_estimate"""
                        conditioning_frame = data[:, t]
                        target_frame = data[:, t + 1]
                    
                        if t < data.shape[1] - 2:
                            noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)

                            # Obtain EOS estimate for next AR step
                            #_, _, x0_estimate = model(conditioning_frame, target_frame, return_x0_estimate=True, limit_timesteps_training=train_params["first_ar_step_noising_step_limit"])
                        else:
                            noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)

                        loss += criterion(predicted_noise, noise)

                    running_val_loss += loss.item()
                
            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- Base Logging Logic ---
        log_dict = {
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "best_traj_time": best_traj_time,
            "best_traj_error": best_traj_time
        }
        
        log_dict = traj_eval_step(traj_loader, epoch, epoch_sampling_frequency, model, device, all_configs, log_dict, checkpoint_dir)

        wandb.log(log_dict)

        print(f"Epoch [{epoch+1}/{train_params['num_epochs']}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}")


    print("Training complete!")
    return model


def train_diffusion_model_initial_exploration(model, train_loader, val_loader, train_params, criterion, all_configs, checkpoint_dir):
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
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr_end)

    print(f"Starting training on {device} for {num_epochs} epochs.")
    print(f"Schedule: Cosine {lr_start:.1e} -> {lr_end:.1e} (Tau={tau})")

    # Storage for results
    eb_free_error = {}  # Maps alpha -> clean error
    eb_free_checkpoints = {} # Maps alpha -> checkpoint path

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
            noise, predicted_noise = model(conditioning_frame, target_frame)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # ==========================
        # 2. Validation & Pruning
        # ==========================
        avg_val_loss = 0.0
        
        # Lists to track what happens in THIS epoch
        steps_to_delete = [] 
        pruned_noise_levels = [] 
        
        if val_loader is not None:
            # --- A. Standard Validation Loss ---
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data_val = sample_val["data"].to(device)
                    if data_val.shape[1] != 2: continue
                    
                    noise, predicted_noise = model(data_val[:, 0], data_val[:, 1])
                    loss_val = criterion(predicted_noise, noise)
                    running_val_loss += loss_val.item()
            avg_val_loss = running_val_loss / (batch_idx_val + 1)

            # --- B. Pruning Check (Single Batch) ---
            check_batch = next(iter(val_loader))
            data_check = check_batch["data"].to(device)
            cond_check = data_check[:, 0]
            target_check = data_check[:, 1]
            
            with torch.no_grad():
                current_timesteps = range(model.timesteps)
                
                for ts in current_timesteps:
                    second_est, first_est = model(cond_check, target_check, 
                                                  fixed_timestep=ts, 
                                                  input_type="input-own-pred")
                    
                    clean_error = criterion(first_est, target_check).item()
                    own_pred_error = criterion(second_est, target_check).item()
                    
                    ratio = own_pred_error / clean_error
                    
                    if ratio < tau:
                        nl_val = model.sqrtOneMinusAlphasCumprod.ravel()[ts].item()
                        print(nl_val, ratio, clean_error, own_pred_error)
                        # Store error immediately
                        eb_free_error[nl_val] = clean_error
                        
                        # Queue for deletion and saving, but don't save yet
                        steps_to_delete.append(ts)
                        pruned_noise_levels.append(nl_val)

            # --- C. Consolidated Saving & Deletion ---
            if steps_to_delete:
                # 1. Save ONE checkpoint for all steps pruned in this epoch
                ckpt_name = f"pruned_epoch_{epoch}.pt"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                torch.save(model.unet.state_dict(), ckpt_path)
                
                # 2. Update the dictionary to point multiple alphas to the same file
                for nl in pruned_noise_levels:
                    eb_free_checkpoints[nl] = ckpt_path

                print(f"--> Pruning {len(steps_to_delete)} steps at Epoch {epoch+1}. Saved to {ckpt_name}")
                print(eb_free_checkpoints)

                map_path = os.path.join(checkpoint_dir, 'checkpoint_map.json')
                with open(map_path, 'w') as f:
                    json.dump(collections.OrderedDict(sorted(eb_free_checkpoints.items())), f, indent=4)
                print(f"Checkpoint map saved to {map_path}")

                map_path = os.path.join(checkpoint_dir, 'error_map.json')
                with open(map_path, 'w') as f:
                    json.dump(collections.OrderedDict(sorted(eb_free_error.items())), f, indent=4)
                print(f"Checkpoint map saved to {map_path}")

                # 3. Delete steps
                model.delete_steps(steps_to_delete)

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
    tau = 1.02
    
    model.to(device)

    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    t_max = train_params.get("T_max", num_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr_end)

    print(f"Starting training on {device} for {num_epochs} epochs.")
    print(f"Schedule: Cosine {lr_start:.1e} -> {lr_end:.1e} (Tau={tau})")

    # Storage for results
    eb_free_error = {}  # Maps alpha -> clean error
    eb_free_checkpoints = {} # Maps alpha -> checkpoint path

    current_timestep = model.timesteps-1

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
                second_est, first_est = model(cond_check, target_check, 
                                                fixed_timestep=current_timestep, 
                                                input_type="input-own-pred")
                
                clean_error = criterion(first_est, target_check).item()
                own_pred_error = criterion(second_est, target_check).item()
                
                ratio = own_pred_error / clean_error
                
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