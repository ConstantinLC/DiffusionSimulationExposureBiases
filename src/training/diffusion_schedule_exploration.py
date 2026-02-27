import os
import torch
import torch.optim as optim
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..utils.diffusion import evaluate_dw_train_inf_gap
from ..utils.general import traj_eval_step

def train_diffusion_single_noise_level(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir, device, is_master):
    """
    Trains a diffusion model on a single noise level with DDP support.
    Breaks training if (own_prediction_error / clean_prediction_error) < tau.
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    tau = train_params["tau"]  # Threshold for early stopping

    # Get world size for averaging metrics
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_params["T_max"], eta_min=lr_end)

    success = False
    best_traj_error = float('inf')
    best_traj_time = 0

    if is_master:
        print(f"Starting Single Level Training on {device}.")
        print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")
        print(f"Stop Threshold (Tau): {tau}")

    for epoch in range(num_epochs):
        # Enforce LR floor manually if epoch exceeds T_max schedule
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
            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]

            optimizer.zero_grad()

            # Forward pass
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

                    noise, predicted_noise = model(conditioning_frame_val, target_frame_val)
                    loss_val = criterion(predicted_noise, noise)
                    running_val_loss += loss_val.item()
            model.train()

            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- Scheduler Step & Get Current LR ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Trajectory Evaluation (same scheme as train_diffusion_model) ---
        model_to_eval = model.module if isinstance(model, DDP) else model

        log_dict = {
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "learning_rate": current_lr,
            "best_traj_time": best_traj_time,
            "best_traj_error": best_traj_error,
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

        # --- Ratio Check (own-prediction vs clean-prediction) ---
        model_to_eval.eval()
        evals = evaluate_dw_train_inf_gap(
            {'model': model_to_eval},
            val_loader,
            device=device,
            n_batches=20,
            metric='mse',
            input_types=['clean', 'own-pred']
        )

        local_clean_err = evals['mse_clean']['model'].float().mean().to(device)
        local_own_err = evals['mse_own_pred']['model'].float().mean().to(device)

        if dist.is_initialized():
            dist.all_reduce(local_clean_err, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_own_err, op=dist.ReduceOp.SUM)
            local_clean_err = local_clean_err / world_size
            local_own_err = local_own_err / world_size

        ratio = local_own_err / local_clean_err

        log_dict["ratio"] = ratio.item()
        log_dict["clean_err"] = local_clean_err.item()
        log_dict["own_err"] = local_own_err.item()

        # --- Logging (Master Only) ---
        if is_master:
            if 'best_traj_time' in log_dict:
                best_traj_time = log_dict['best_traj_time']
            if 'best_traj_error' in log_dict:
                best_traj_error = log_dict['best_traj_error']

            wandb.log(log_dict)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"Ratio: {ratio:.4f} (Tau: {tau}) | LR: {current_lr:.2e} | "
                  f"Clean error: {local_clean_err} | "
                  f"Own error: {local_own_err}")

        # Sync all processes after logging/eval
        if isinstance(model, DDP):
            dist.barrier()

        # --- Break Condition ---
        if ratio < tau:
            success = True
            if is_master:
                print(f" >>> SUCCESS: Ratio {ratio:.4f} < {tau}. Converged.")
            break

    if is_master:
        print("Training complete!")

    return model, success
