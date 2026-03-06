import os
import json
import inspect
import torch
from torch import nn
import wandb
from torch.nn import functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config import ExperimentConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel
from src.training.diffusion_schedule_exploration import train_diffusion_single_noise_level
from src.utils.general import count_parameters, get_run_dir_name
from src.utils.diffusion import evaluate_dw_train_inf_gap
from src.utils.multigpu import setup_ddp, cleanup


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_cfg_raw = OmegaConf.to_container(cfg.model, resolve=True)
    model_class_name = model_cfg_raw.get("class", "DiffusionModel")

    if model_class_name != "DiffusionModel":
        raise ValueError(
            f"train_diffusion_initial_exploration only supports DiffusionModel, "
            f"got '{model_class_name}'"
        )

    # Select training params from pretraining/finetuning based on checkpoint presence
    has_checkpoint = bool(model_cfg_raw.get("checkpoint", ""))
    training_source = cfg.finetuning if has_checkpoint else cfg.pretraining
    mode = "finetuning" if has_checkpoint else "pretraining"
    print(f"Using {mode} training parameters.")
    cfg = OmegaConf.merge(cfg, {"training": OmegaConf.to_container(training_source, resolve=True)})

    config = ExperimentConfig.from_hydra(cfg)

    # --- DDP setup (activated by torchrun) ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        import torch.distributed as dist
        setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        is_master = local_rank == 0
    else:
        device = torch.device(config.training.device)
        is_master = True

    # --- Checkpoint directory (master only to avoid race) ---
    run_name = "debug"
    checkpoint_dir = None
    if is_master:
        run_name = get_run_dir_name(config.checkpoint_dir, config.model)
        run_name = run_name + "_binary_search"
        if not config.debugging:
            checkpoint_dir = os.path.join(config.checkpoint_dir, run_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Artifacts for this run will be saved in: {checkpoint_dir}")
        else:
            print("Debugging mode enabled: no checkpoint directory will be created.")

    if is_distributed:
        sync = [run_name, checkpoint_dir]
        dist.broadcast_object_list(sync, src=0)
        dist.barrier()
        run_name, checkpoint_dir = sync[0], sync[1]

    # --- Save config ---
    legacy = config.to_legacy_dict()
    legacy["model_params"]["class"] = model_class_name
    if is_master and checkpoint_dir is not None:
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(legacy, f, indent=4)

    # --- Loss function ---
    if config.loss.name == "mse":
        criterion = nn.MSELoss()
    elif config.loss.name == "l1":
        criterion = F.smooth_l1_loss
    else:
        raise ValueError(f"Unknown loss: {config.loss.name}")

    # --- Binary Search Parameters ---
    low_log_sigma = -4.0
    high_log_sigma = -0.5
    epsilon = 0.1
    tau = config.training.tau

    best_passing_sigma = None

    if is_master:
        print(f"--- Starting Binary Search ---")
        print(f"Range: [{low_log_sigma}, {high_log_sigma}] | Target Precision: {epsilon} | Tau: {tau}")

    # Build initial model (schedule will be overridden each iteration)
    _valid_model_params = inspect.signature(DiffusionModel.__init__).parameters
    model_params = {k: v for k, v in legacy["model_params"].items() if k in _valid_model_params}
    model = DiffusionModel(**model_params)
    model.to(device)

    if is_master:
        print(f"Parameters: {count_parameters(model):,}")

    prev_checkpoint_path = None
    iteration = 0

    while (high_log_sigma - low_log_sigma) > epsilon:
        iteration += 1
        mid_log_sigma = (low_log_sigma + high_log_sigma) / 2
        sigma_val = torch.tensor([10 ** mid_log_sigma])

        if is_master:
            print(f"\n=== ITERATION {iteration} ===")
            print(f"Testing Log Sigma: {mid_log_sigma:.4f} (Val: {sigma_val.item():.5f})")
            print(f"Current Interval: [{low_log_sigma:.4f}, {high_log_sigma:.4f}]")

        # Per-iteration checkpoint sub-folder
        iter_checkpoint_dir = None
        if is_master and checkpoint_dir is not None:
            iter_checkpoint_dir = os.path.join(
                checkpoint_dir, f"iter_{iteration}_sigma_{mid_log_sigma:.3f}"
            )
            os.makedirs(iter_checkpoint_dir, exist_ok=True)

        if is_distributed:
            sync = [iter_checkpoint_dir]
            dist.broadcast_object_list(sync, src=0)
            iter_checkpoint_dir = sync[0]

        # Load checkpoint from previous iteration into the bare model
        raw_model = model.module if isinstance(model, DDP) else model
            
        # Test without using checkpoint first
        prev_checkpoint_path = None

        if prev_checkpoint_path is not None:
            if is_master:
                print(f"Loading checkpoint from previous iteration: {prev_checkpoint_path}")
            state_dict = torch.load(prev_checkpoint_path, map_location=device)
            raw_model.load_state_dict(state_dict)

        # Reset schedule for this iteration
        raw_model.compute_schedule_variables(sigmas=sigma_val.to(device))

        # Wrap in DDP once (stays wrapped for subsequent iterations)
        if is_distributed and not isinstance(model, DDP):
            model = DDP(model, device_ids=[local_rank])

        # W&B run for this iteration
        if is_master:
            wandb.init(
                project=config.wandb.project + ("_sr" if config.data.super_resolution else ""),
                entity=config.wandb.entity,
                name=f"{run_name}_iter{iteration}",
                group=run_name,
                config={**legacy, "log_sigma": mid_log_sigma},
                mode="disabled" if config.debugging else "online",
                reinit=True,
            )

        # Build fresh data loaders for this iteration (same as train.py)
        train_loader, val_loader, traj_loader = get_data_loaders(
            config.data, is_distributed=is_distributed
        )

        # Train
        _, iter_success = train_diffusion_single_noise_level(
            model,
            train_loader,
            val_loader,
            traj_loader,
            legacy["train_params"],
            criterion,
            legacy,
            iter_checkpoint_dir,
            device=device,
            is_master=is_master,
        )

        # Save checkpoint for the next iteration
        if is_master and iter_checkpoint_dir is not None:
            prev_checkpoint_path = os.path.join(iter_checkpoint_dir, "iter_final.pth")
            torch.save(raw_model.state_dict(), prev_checkpoint_path)
            print(f"Saved iteration checkpoint: {prev_checkpoint_path}")

        if is_distributed:
            sync = [prev_checkpoint_path]
            dist.broadcast_object_list(sync, src=0)
            prev_checkpoint_path = sync[0]

        # Evaluate success: own-prediction ratio vs. clean-prediction error
        raw_model.eval()
        evals = evaluate_dw_train_inf_gap(
            {"model": raw_model},
            val_loader,
            n_batches=10,
            metric="mse",
            device=device,
            input_types=["clean", "own-pred"],
        )
        clean_err = evals["mse_clean"]["model"].float().mean().to(device)
        own_err = evals["mse_own_pred"]["model"].float().mean().to(device)

        if is_distributed:
            dist.all_reduce(clean_err, op=dist.ReduceOp.SUM)
            dist.all_reduce(own_err, op=dist.ReduceOp.SUM)
            clean_err = clean_err / dist.get_world_size()
            own_err = own_err / dist.get_world_size()

        ratio = own_err / clean_err
        success = ratio.item() < tau

        if is_master:
            print(f"Clean error: {clean_err.item()}")
            print(f"Own error: {own_err.item()}")
            print(f"Ratio: {ratio.item():.4f} (tau={tau}) -> {'PASSED' if success else 'FAILED'}")
            wandb.log({
                "iter": iteration,
                "mid_log_sigma": mid_log_sigma,
                "ratio": ratio.item(),
                "success": int(success),
            })
            wandb.finish()

        # Binary search update
        if success:
            if is_master:
                print("Result: PASSED. Searching lower interval...")
            best_passing_sigma = mid_log_sigma
            high_log_sigma = mid_log_sigma
        else:
            if is_master:
                print("Result: FAILED. Searching higher interval...")
            low_log_sigma = mid_log_sigma

    # --- Final Result ---
    if is_master:
        print("\n" + "=" * 40)
        print("BINARY SEARCH COMPLETE")
        print("=" * 40)
        if best_passing_sigma is not None:
            print(f"Smallest viable Log Sigma: {best_passing_sigma:.5f}")
            print(f"Value: {10 ** best_passing_sigma:.5f}")
        else:
            print("No viable sigma found in the range (all failed).")

    if is_distributed:
        cleanup()


if __name__ == "__main__":
    main()
