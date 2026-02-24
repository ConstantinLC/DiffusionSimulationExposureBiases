import os
import json
import torch
from torch import nn
import wandb
from torch.nn import functional as F
import hydra
from omegaconf import DictConfig

from src.config import ExperimentConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel
from src.training.diffusion_trainer import train_diffusion_model, train_diffusion_model_multisteps
from src.utils.general import count_parameters, get_next_run_number
from src.utils.diffusion import betas_from_sqrtOneMinusAlphasCumprod
from src.utils.multigpu import setup_ddp, cleanup


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    config = ExperimentConfig.from_hydra(cfg)

    device = torch.device(config.training.device)

    # --- Setup checkpoint directory ---
    run_number = get_next_run_number(config.checkpoint_dir)
    checkpoint_dir = os.path.join(config.checkpoint_dir, f'run_{run_number}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Artifacts for this run will be saved in: {checkpoint_dir}")

    # Save config in checkpoint folder
    legacy = config.to_legacy_dict()
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(legacy, f, indent=4)

    # --- Initialize W&B ---
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=f"run_{run_number}",
        config=legacy
    )

    # --- Initialize model and data ---
    train_loader, val_loader, traj_loader = get_data_loaders(config.data)
    model = DiffusionModel(**legacy['model_params'])
    model.to(device)

    print(f"Model has {count_parameters(model)} parameters.")

    # Initialize loss function
    if config.loss.name == 'mse':
        print("Using MSELoss")
        criterion = nn.MSELoss()
    elif config.loss.name == 'l1':
        print("Using L1-Loss")
        criterion = F.smooth_l1_loss
    else:
        raise ValueError(f"Unknown loss function name: {config.loss.name}")

    # Start training
    if config.data.prediction_steps == 1:
        trained_model = train_diffusion_model(
            model,
            train_loader,
            val_loader,
            traj_loader,
            legacy['train_params'],
            criterion,
            legacy,
            checkpoint_dir,
            device=device,
            is_master=True
        )
    else:
        trained_model = train_diffusion_model_multisteps(
            model,
            train_loader,
            val_loader,
            traj_loader,
            legacy['train_params'],
            criterion,
            legacy,
            checkpoint_dir,
            device=device,
            is_master=True
        )

    wandb.finish()


if __name__ == '__main__':
    main()
