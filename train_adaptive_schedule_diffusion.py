import os
import json
import argparse
import torch
from torch import nn
import wandb
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.trainer import train_diffusion_model
from src.utils import count_parameters
from src.utils import get_next_run_number
from torch.nn import functional as F
from src.diffusion_utils import betas_from_sqrtOneMinusAlphasCumprod, evaluate_dw_train_inf_gap, cosine_sigma_schedule

def adapt_schedule(noise_levels, own_pred_errors, prev_pred_errors, clean_errors, tau, log_incr, device='cuda'):
    # Ensure inputs are on CPU for logic processing
    noise_levels = noise_levels
    own_pred_errors = own_pred_errors
    prev_pred_errors = prev_pred_errors
    clean_errors = clean_errors

    own_ratio = own_pred_errors / clean_errors
    prev_ratio = prev_pred_errors / clean_errors

    finetune_needed = False

    T = len(noise_levels)
    indent = 0
    if own_ratio[0] > tau:
        noise_levels[0] *= 10**log_incr
        finetune_needed = True

    for i in range(1, T-1):
        new_level = None
        idx = i + indent
        if noise_levels[idx] < noise_levels[0]:
            noise_levels[idx] = noise_levels[0]
        elif noise_levels[idx] != noise_levels[0]:
            if own_ratio[i] > tau:
                new_level = noise_levels[idx]
                finetune_needed = True

            elif prev_ratio[i] > tau:
                if i < T - 1:
                    noise_levels[idx] = (noise_levels[idx] + noise_levels[idx + 1]) / 2
                    finetune_needed = True

            if new_level is not None:
                noise_levels = torch.cat((noise_levels[:idx+1], torch.tensor([new_level], device=device), noise_levels[idx+1:]))
                indent += 1

    return noise_levels, indent, finetune_needed

def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument('--config', type=str, default='configs/config.json',
                        help='Path to configuration JSON file.')
    args = parser.parse_args()

    # Initialize DDP
    local_rank, global_rank = setup_ddp()
    is_master = (global_rank == 0)
    device = torch.device("cuda", local_rank)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    if is_master:
        print(f"Loaded config from: {args.config}")

    # --- Setup checkpoint directory (Master Only) ---
    if is_master:
        base_checkpoint_dir = config["data_params"]["base_checkpoint_dir"]
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
    else:
        checkpoint_dir = None
        run_number = None

    # --- Initialize model and data ---
    train_loader, val_loader, traj_loader = get_data_loaders(config['data_params'], is_distributed=True)
    model = DiffusionModel(**config['model_params'])
    model.to(device)

    if is_master:
        print(f"Model has {count_parameters(model)} parameters.")

    # Initialize loss function
    if config['loss_params']['name'] == 'mse':
        if is_master:
            print("Using MSELoss")
        criterion = nn.MSELoss()
    elif config['loss_params']['name'] == 'l1':
        if is_master:
            print("Using L1-Loss")
        criterion = F.smooth_l1_loss
    else:
        raise ValueError(f"Unknown loss function name: {config['loss_params']['name']}")
    
    tau = 1.05

    #### BINARY SEARCH ####

    minimal_log_sigma = -5
    maximal_log_sigma = -1
    curr_log_sigma = (maximal_log_sigma + minimal_log_sigma)/2

    if config["train_params"]["best_log_sigma"] is not None:
        best_sigma = config["train_params"]["best_log_sigma"]

    if config["train_params"]["perform_binary_search"]:

        best_sigma = -1

        for i in range(config["train_params"]["binary_search_steps"]):

            # Synchronize processes before new iteration
            dist.barrier()

            sampling_sigmas = cosine_sigma_schedule(10**curr_log_sigma, 10**-0.0001, 20).to(device)
            training_sigmas = torch.concatenate((torch.ones(80, device=device)*sampling_sigmas[0], sampling_sigmas))
            model.compute_schedule_variables(sigmas=training_sigmas)

            # Wrap with DDP
            if not isinstance(model, DDP):
                model = DDP(model, device_ids=[local_rank])

            trained_model = train_diffusion_model(
                model,
                train_loader,
                val_loader,
                traj_loader,
                config['train_params'],
                criterion,
                config,
                checkpoint_dir,
                device=device,
                is_master=is_master
            )

            model_name = f"{curr_log_sigma}"

            model.compute_schedule_variables(sigmas=sampling_sigmas)
            results = evaluate_dw_train_inf_gap({model_name:trained_model}, val_loader, device=device.type)

            if is_master:
                torch.save(model.state_dict(), os.path.join(f"{checkpoint_dir}", f"binary_search_iteration{i}.pth"))

            mse_clean = results['mse_clean'][model_name]
            mse_clean_own_pred = results['mse_clean_own_pred'][model_name]
            mse_clean_prev_pred = results['mse_clean_prev_pred'][model_name]

            own_prediction_bias = mse_clean_own_pred[-1] / mse_clean[-1]

            if  own_prediction_bias < tau:
                if is_master:
                    print(f"Log-Sigma {curr_log_sigma} under Instability Threshold !, tau = {own_prediction_bias}")
                if curr_log_sigma < best_sigma:
                    best_sigma = curr_log_sigma
                maximal_log_sigma = curr_log_sigma
            else:
                if is_master:
                    print(f"Log-Sigma {curr_log_sigma} too Low!, tau = {own_prediction_bias}")
                minimal_log_sigma = curr_log_sigma

            curr_log_sigma = (maximal_log_sigma + minimal_log_sigma)/2

        if is_master:
            print(f"Final Sigma: {best_sigma}")

    #### FINETUNING ####

    log_incr = 0.1

    sampling_sigmas = cosine_sigma_schedule(10**best_sigma, 10**-0.0001, 20).to(device)
    training_sigmas = torch.concatenate(((torch.ones(80, device=device)*sampling_sigmas[0]), sampling_sigmas))
    model.compute_schedule_variables(sigmas=training_sigmas)

    training_T = 100
    sampling_T = 20

    for i in range(config["train_params"]["finetuning_steps"]):

        # Synchronize processes
        dist.barrier()

        if is_master:
            print(sampling_sigmas)

        model = train_diffusion_model(
            model,
            train_loader,
            val_loader,
            traj_loader,
            config['train_params'],
            criterion,
            config,
            checkpoint_dir,
            device=device,
            is_master=is_master
        )

        model_name = "curr"
        model.compute_schedule_variables(sigmas=sampling_sigmas)
        results = evaluate_dw_train_inf_gap({model_name:model}, val_loader, device=device.type)

        if is_master:
            torch.save(model.state_dict(), os.path.join(f"{checkpoint_dir}", f"finetuning_iteration{i}.pth"))

        mse_clean = results['mse_clean'][model_name].flip([0])
        mse_clean_own_pred = results['mse_clean_own_pred'][model_name].flip([0])
        mse_clean_prev_pred = results['mse_clean_prev_pred'][model_name].flip([0])

        sampling_sigmas, added_T, finetuned_needed = adapt_schedule(sampling_sigmas, mse_clean_own_pred, mse_clean_prev_pred, mse_clean, tau, log_incr, device)
        sampling_T += added_T

        if not finetuned_needed :
            if is_master:
                print(f"Finetuning done at iteration {i}")
            break
        else:
            training_sigmas = torch.concatenate((sampling_sigmas[0]*torch.ones(training_T-sampling_T, device=device), sampling_sigmas))
            model.compute_schedule_variables(sigmas=training_sigmas)

    if is_master and finetuned_needed:
        print("Finetuning done, but some transitions still have a bias above tau..")
        wandb.finish()

    cleanup()

if __name__ == '__main__':
    main()