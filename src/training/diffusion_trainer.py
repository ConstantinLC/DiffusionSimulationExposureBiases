import torch
import torch.optim as optim
import numpy as np
import wandb
import os
from ..utils.general import traj_eval_step
from ..utils.diffusion import compute_estimate, compute_sigmas_refiner, evaluate_dw_train_inf_gap
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import collections
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


TAU_THRESHOLDS = [1.0, 1.05, 1.1, 1.15]


def update_error_tracking_map(error_tracking_map, model, val_loader, device, epoch,
                              tau_thresholds=None, n_batches=20, validate_every_k=10,
                              n_noise_samples=1):
    """
    For each noise level sigma and each tau threshold, record the first epoch
    where the mean per-sample ratio mean(E_own_i / E_clean_i) drops below tau,
    storing the mean E_clean at that point as gamma(sigma, tau).

    n_noise_samples: number of independent noise draws per batch per noise level.
                     Results are averaged before accumulation.
    """
    if tau_thresholds is None:
        tau_thresholds = TAU_THRESHOLDS

    if epoch % validate_every_k != 0:
        return error_tracking_map

    model.eval()

    if hasattr(model, 'sqrtOneMinusAlphasCumprod'):
        sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()  # index 0 = sigma_min, matches x0_estimates order
    else:
        sigmas = torch.linspace(0, 1, model.timesteps)
    T = len(sigmas)

    # Accumulate per-sample values across batches: lists of length T
    all_clean = [[] for _ in range(T)]
    all_ratio = [[] for _ in range(T)]

    spatial_dims = None

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            if batch_idx >= n_batches:
                break

            data   = sample['data'].to(device)
            cond   = data[:, 0]
            target = data[:, 1]

            if spatial_dims is None:
                spatial_dims = tuple(range(1, target.ndim))

            # Accumulate over multiple noise draws, then average
            sum_clean_mse = None  # (T, B)
            sum_own_mse   = None  # (T, B)
            for _ in range(n_noise_samples):
                _, ests_clean = model(conditioning=cond, data=target,
                                      return_x0_estimate=True, input_type='clean')
                _, ests_own   = model(conditioning=cond, data=target,
                                      return_x0_estimate=True, input_type='own-pred')

                batch_clean = torch.stack(
                    [(e - target).pow(2).mean(dim=spatial_dims) for e in ests_clean]
                )  # (T, B)
                batch_own = torch.stack(
                    [(e - target).pow(2).mean(dim=spatial_dims) for e in ests_own]
                )  # (T, B)

                sum_clean_mse = batch_clean if sum_clean_mse is None else sum_clean_mse + batch_clean
                sum_own_mse   = batch_own   if sum_own_mse   is None else sum_own_mse   + batch_own

            avg_clean_mse = sum_clean_mse / n_noise_samples  # (T, B)
            avg_own_mse   = sum_own_mse   / n_noise_samples  # (T, B)

            for t_idx in range(T):
                clean_mse_i = avg_clean_mse[t_idx]
                own_mse_i   = avg_own_mse[t_idx]

                valid = clean_mse_i > 0
                ratio_i = (own_mse_i / clean_mse_i.clamp(min=1e-12))[valid]
                clean_i = clean_mse_i[valid]

                all_clean[t_idx].append(clean_i.cpu())
                all_ratio[t_idx].append(ratio_i.cpu())

    # Compute stats over all accumulated samples, then check thresholds
    for t_idx in range(T):
        if not all_clean[t_idx]:
            continue

        clean_all = torch.cat(all_clean[t_idx])
        ratio_all = torch.cat(all_ratio[t_idx])

        mean_ratio = ratio_all.mean().item()
        mean_clean = clean_all.mean().item()

        min_idx = ratio_all.argmin().item()
        max_idx = ratio_all.argmax().item()

        sigma_str = str(sigmas[t_idx].item())
        if sigma_str not in error_tracking_map:
            error_tracking_map[sigma_str] = {}

        for tau in tau_thresholds:
            tau_str = str(tau)
            if tau - 0.01 < mean_ratio < tau + 0.01:
                if tau_str not in error_tracking_map[sigma_str]:
                    error_tracking_map[sigma_str][tau_str] = {
                        'clean_error':   mean_clean,
                        'ratio':         mean_ratio,
                        'epoch':         epoch,
                        'min_bias':      ratio_all[min_idx].item(),
                        'min_bias_clean': clean_all[min_idx].item(),
                        'max_bias':      ratio_all[max_idx].item(),
                        'max_bias_clean': clean_all[max_idx].item(),
                    }
                break  # only record the smallest unrecorded threshold per epoch

    return error_tracking_map

def train_diffusion_model(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir, device, is_master, track_instability=False, n_noise_samples=1, validate_every_k=10):
    """
    Trains a diffusion model with Cosine Annealing Learning Rate and DDP support.

    Args:
        track_instability: if True, compute per-noise-level own-prediction bias
            at each validation epoch and save an error_tracking_map.json to
            checkpoint_dir.  This is used to produce Figure 2 (instability
            thresholds) in the paper.
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]

    # --- Config extraction ---
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]

    # Note: Model is already moved to device and wrapped in DDP in main.py.
    # We use the passed 'device' for data transfers.

    # --- Error tracking map for instability threshold validation ---
    error_tracking_map = {}

    # --- 1. Optimizer & Scheduler Setup ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    
    # T_max is the number of epochs until the LR reaches the minimum
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_params["T_max"], 
        eta_min=lr_end
    )

    if is_master:
        print(f"Starting Diffusion training on {device} for {num_epochs} epochs...")
        print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")

    best_traj_error = float('inf')
    best_traj_time = 0

    for epoch in range(num_epochs):
        # Enforce LR floor manually if epoch exceeds T_max schedule
        if epoch >= train_params["T_max"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_end

        # --- DDP: Set Epoch for Sampler ---
        # Crucial for shuffling to work correctly across GPUs in DistributedSampler
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- Training Loop ---
        if not train_params["end_to_end"]:
            model.train()
        else:
            model.eval()

        

        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):

            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]
            
            optimizer.zero_grad()
            if not train_params["end_to_end"]:
                # Diffusion specific forward pass
                noise, predicted_noise = model(conditioning_frame, target_frame)
                loss = criterion(predicted_noise, noise)
            
            else:
                _, noises, predicted_noises  = model(conditioning_frame, target_frame,
                                                    return_noise_pred = True,
                                                    input_type="own-pred")
                loss = criterion(torch.concatenate(noises), torch.concatenate(predicted_noises))

            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # --- Validation Loop ---
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            # If using DistributedSampler for validation, set epoch
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

                    # Diffusion specific validation pass
                    if not train_params["end_to_end"]:
                        # Diffusion specific forward pass
                        noise, predicted_noise = model(conditioning_frame_val, target_frame_val)
                        loss_val = criterion(predicted_noise, noise)
                    else:
                        _, noises, predicted_noises  = model(conditioning_frame_val, target_frame_val,
                                                    return_noise_pred = True,
                                                    input_type="own-pred")
                        loss_val = criterion(torch.concatenate(noises), torch.concatenate(predicted_noises))
                        
                    running_val_loss += loss_val.item()
            model.train()

            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- 2. Step Scheduler & Get Current LR ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Trajectory Evaluation (all processes participate to avoid barrier timeout) ---
        # Use the underlying model (.module) so checkpoint keys don't have "module." prefix,
        # and so inference doesn't trigger DDP gradient-sync machinery.
        model_to_eval = model.module if isinstance(model, DDP) else model

        log_dict = {
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "learning_rate": current_lr,
            "best_traj_time": best_traj_time,
            "best_traj_error": best_traj_error
        }

        log_dict = traj_eval_step(
            traj_loader,
            epoch,
            epoch_sampling_frequency,
            model_to_eval,
            device,
            all_configs,
            log_dict,
            checkpoint_dir,
            is_master=is_master,
        )

        # --- Instability Threshold Tracking ---
        # Run every epoch (independent of traj eval frequency) to get dense coverage.
        if track_instability:
            error_tracking_map = update_error_tracking_map(
                error_tracking_map, model_to_eval, val_loader, device, epoch + 1,
                n_noise_samples=n_noise_samples,
                validate_every_k=validate_every_k,
            )
            if is_master and checkpoint_dir is not None:
                with open(os.path.join(checkpoint_dir, "error_tracking_map.json"), "w") as f:
                    json.dump(error_tracking_map, f, indent=4)

        # --- Logging (Master Only) ---
        if is_master:
            # Update local bests
            if 'best_traj_time' in log_dict:
                best_traj_time = log_dict['best_traj_time']
            if 'best_traj_error' in log_dict:
                best_traj_error = log_dict['best_traj_error']

            wandb.log(log_dict)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

        # Sync all processes after logging/eval
        if isinstance(model, DDP):
            import torch.distributed as dist
            dist.barrier()

    if is_master:
        print("Training complete!")
        
    return model



def train_diffusion_model_multisteps(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir):

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

    n_proxy_steps = train_params["n_proxy_steps"]
    backgrad = train_params["backgrad"]

    best_val_loss = float('inf')
    best_traj_time = 0
    best_traj_error = 100000000

    print(f"Starting Multi-steps Diffusion training on {device} for {train_params['num_epochs']} epochs...")

    for epoch in range(num_epochs):
        if epoch >= train_params["T_max"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_end

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

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        # --- Base Logging Logic ---
        log_dict = {
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "learning_rate": current_lr,  # Log LR to visualize the cosine curve
            "best_traj_time": best_traj_time,
            "best_traj_error": best_traj_time
        }
        
        log_dict = traj_eval_step(traj_loader, epoch, epoch_sampling_frequency, model, device, all_configs, log_dict, checkpoint_dir)

        wandb.log(log_dict)

        print(f"Epoch [{epoch+1}/{train_params['num_epochs']}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}")


    print("Training complete!")
    return model