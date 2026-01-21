import torch
import torch.optim as optim
import wandb
import os
from src.utils import evaluate_trajectory
from src.diffusion_utils import compute_estimate, betas_from_sqrtOneMinusAlphasCumprod
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
import json
import collections
from torch.nn import functional as F

import os
import torch

def traj_eval_step(traj_loader, epoch, epoch_sampling_frequency, model, device, all_configs, log_dict, checkpoint_dir):
    """
    Evaluates trajectory metrics using the unified evaluate_trajectory function
    and handles logging and best-model saving.
    """
    # Check if we should evaluate this epoch
    if traj_loader and epoch % epoch_sampling_frequency == 0 and epoch > 0:
        model.eval()
        print("Evaluating on trajectories...")

        # 1. Retrieve configuration
        metrics_list = all_configs["loss_params"]["eval_traj_metrics"]
        primary_metric = all_configs["loss_params"]["primary_metric"]

        # 2. Run the unified evaluation (computes all requested metrics in one pass)
        results = evaluate_trajectory(model, traj_loader, device, metrics=metrics_list)

        # 3. Log detailed per-timestep metrics
        # Map result keys to the log_dict keys you expect
        if 'vort_corr' in metrics_list:
            for t, val in enumerate(results['vort_corr_per_ts']):
                log_dict[f'vorticity_correlation_t_{t+1}'] = val
        
        if 'mse' in metrics_list:
            for t, val in enumerate(results['mse_per_ts']):
                log_dict[f'mse_t_{t+1}'] = val
        
        if 'corr' in metrics_list:
            for t, val in enumerate(results['corr_per_ts']):
                log_dict[f'correlation_t_{t+1}'] = val

        # 4. Handle Best Model Selection
        if primary_metric == "vort_corr" and 'vort_corr' in metrics_list:
            # Metric: Time until correlation drops below threshold
            current_traj_time = results['vort_corr_time_under_threshold']
            log_dict['time_under_0.8'] = current_traj_time
            print(f"Vorticity correlation time under 0.8: {current_traj_time}")

            # Check for improvement (Higher time is better)
            if current_traj_time > log_dict.get('best_traj_time', -1):
                best_traj_time = current_traj_time
                log_dict['best_traj_time'] = best_traj_time
                
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ New best model saved to {best_checkpoint_path} with trajectory time: {best_traj_time}")

        elif primary_metric == "corr" and 'corr' in metrics_list:
            # Metric: Time until correlation drops below threshold
            current_traj_time = results['corr_time_under_threshold']
            log_dict['time_under_0.8'] = current_traj_time
            print(f"Correlation time under 0.8: {current_traj_time}")

            # Check for improvement (Higher time is better)
            if current_traj_time > log_dict.get('best_traj_time', -1):
                best_traj_time = current_traj_time
                log_dict['best_traj_time'] = best_traj_time
                
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ New best model saved to {best_checkpoint_path} with trajectory time: {best_traj_time}")

        elif primary_metric == "mse" and 'mse' in metrics_list:
            # Metric: Mean MSE over trajectory
            current_traj_error = results['mean_mse']
            print(f"📉 Mean Trajectory MSE: {current_traj_error:.2e}")

            # Initialize best_traj_error if missing
            if log_dict.get('best_traj_error') is None:
                log_dict['best_traj_error'] = float('inf')

            # Check for improvement (Lower MSE is better)
            if current_traj_error < log_dict['best_traj_error']:
                best_traj_error = current_traj_error
                log_dict['best_traj_error'] = best_traj_error
                
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ New best model saved to {best_checkpoint_path} with trajectory error: {best_traj_error:.2e}")

        # 5. Routine Checkpoint Saving
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

    #scheduler = StepLR(optimizer, num_epochs//2, gamma=0.1, last_epoch=-1)

    print(f"Starting Diffusion training on {device} for {num_epochs} epochs...")
    print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")

    best_traj_error = float('inf')
    best_traj_time = 0

    for epoch in range(num_epochs):
       # if epoch >= 100:
        #    for param_group in optimizer.param_groups:
         #       param_group['lr'] = 0.00001

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
    lr_start = train_params["learning_rate_start"]
    #lr_end = train_params["learning_rate_end"]

    optimizer = optim.Adam(model.parameters(), lr=lr_start)
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
                target_frame = data[:, t + 1]
            
                if t < data.shape[1] - 2:
                    noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)

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
        
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data = sample["data"].to(device)
                    loss = torch.Tensor([0]).to(device)

                    for t in range(data.shape[1] - 1):
                        conditioning_frame = data[:, t]
                        target_frame = data[:, t + 1]
                    
                        if t < data.shape[1] - 2:
                            noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)
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
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=lr_end)

    #scheduler = StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)

    print(f"Starting training on {device} for {num_epochs} epochs.")
    print(f"Schedule: Cosine {lr_start:.1e} -> {lr_end:.1e} (Tau={tau})")

    # Storage for results
    eb_free_error = {}  # Maps alpha -> clean error
    eb_free_checkpoints = {} # Maps alpha -> checkpoint path

    initial_noise_levels = model.sqrtOneMinusAlphasCumprod.ravel()
    cur_window = 0
    cur_noise_levels = initial_noise_levels[20-(cur_window+1)*5:20-(cur_window)*5]

    print(cur_noise_levels)

    model.compute_schedule_variables(betas_from_sqrtOneMinusAlphasCumprod(cur_noise_levels))

    print(model.sqrtOneMinusAlphasCumprod.ravel())

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
                    
                    ratio = own_pred_error / clean_error

                    print(model.sqrtOneMinusAlphasCumprod.ravel()[ts].item(), ratio, clean_error)
                    
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
                if len(steps_to_delete) == len(model.betas.ravel()):
                    cur_window += 1
                    print(cur_window)
                    cur_noise_levels = initial_noise_levels[20-(cur_window+1)*5:20-(cur_window)*5]
                    model.compute_schedule_variables(betas_from_sqrtOneMinusAlphasCumprod(cur_noise_levels))
                else:
                    model.delete_steps(steps_to_delete)

                #optimizer.state.clear()

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