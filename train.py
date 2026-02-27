import os
import json
import torch
from torch import nn
import wandb
from torch.nn import functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config import ExperimentConfig, DiffusionModelConfig, RefinerConfig, Unet2DConfig, Unet1DConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel
from src.models.pderefiner import PDERefiner
from src.models.unet_2d import Unet
from src.models.unet_1d import Unet1D
from src.training.diffusion_trainer import train_diffusion_model, train_diffusion_model_multisteps
from src.training.unet_trainer import train_unet, train_unet_multisteps
from src.utils.general import count_parameters, get_run_dir_name
from src.utils.multigpu import setup_ddp, cleanup


def build_model(config: ExperimentConfig) -> nn.Module:
    """Instantiate the correct model class from the typed model config."""
    mp = config.model

    if isinstance(mp, DiffusionModelConfig):
        return DiffusionModel(
            dimension=mp.dimension,
            dataSize=mp.dataSize,
            condChannels=mp.condChannels,
            dataChannels=mp.dataChannels,
            diffSchedule=mp.diffSchedule,
            diffSteps=mp.diffSteps,
            inferenceSamplingMode=mp.inferenceSamplingMode,
            inferenceConditioningIntegration=mp.inferenceConditioningIntegration,
            diffCondIntegration=mp.diffCondIntegration,
            padding_mode=mp.padding_mode,
            architecture=mp.architecture,
            checkpoint=mp.checkpoint,
            load_betas=mp.load_betas,
        )

    elif isinstance(mp, RefinerConfig):
        return PDERefiner(
            dimension=mp.dimension,
            dataSize=mp.dataSize,
            condChannels=mp.condChannels,
            dataChannels=mp.dataChannels,
            refinementSteps=mp.refinementSteps,
            log_sigma_min=mp.log_sigma_min,
            padding_mode=mp.padding_mode,
            architecture=mp.architecture,
            checkpoint=mp.checkpoint,
        )

    elif isinstance(mp, Unet2DConfig):
        return Unet(
            dim=mp.dim if mp.dim is not None else mp.dataSize[0],
            sigmas=torch.zeros(1),
            channels=mp.condChannels,
            dim_mults=tuple(mp.dim_mults),
            use_convnext=True,
            convnext_mult=mp.convnext_mult,
            padding_mode=mp.padding_mode,
            with_time_emb=mp.with_time_emb,
        )

    elif isinstance(mp, Unet1DConfig):
        return Unet1D(
            dim=mp.dim if mp.dim is not None else mp.dataSize[0],
            sigmas=torch.tensor(1),
            channels=mp.condChannels,
            dim_mults=tuple(mp.dim_mults),
            use_convnext=True,
            convnext_mult=mp.convnext_mult,
            padding_mode=mp.padding_mode,
            with_time_emb=mp.with_time_emb,
        )

    else:
        raise ValueError(f"Unknown model config type: {type(mp)}")


def _make_criterion(config: ExperimentConfig):
    if config.loss.name == "mse":
        print("Using MSELoss")
        return nn.MSELoss()
    elif config.loss.name == "l1":
        print("Using smooth-L1 loss")
        return F.smooth_l1_loss
    else:
        raise ValueError(f"Unknown loss: {config.loss.name}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Select training params from pretraining/finetuning based on checkpoint presence
    has_checkpoint = bool(OmegaConf.select(cfg, "model.checkpoint", default=""))
    training_source = cfg.finetuning if has_checkpoint else cfg.pretraining
    mode = "finetuning" if has_checkpoint else "pretraining"
    print(f"Using {mode} training parameters.")
    cfg = OmegaConf.merge(cfg, {"training": OmegaConf.to_container(training_source, resolve=True)})
    
    config = ExperimentConfig.from_hydra(cfg)
    model_type = config.model.type
    print(config)

    # --- DDP setup (activated by torchrun) ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
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
        if not config.debugging:
            checkpoint_dir = os.path.join(config.checkpoint_dir, run_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Artifacts for this run will be saved in: {checkpoint_dir}")
        else:
            print(f"Debugging mode enabled: no checkpoint directory will be created.")
    if is_distributed:
        import torch.distributed as dist
        sync = [run_name, checkpoint_dir]
        dist.broadcast_object_list(sync, src=0)
        dist.barrier()
        run_name, checkpoint_dir = sync[0], sync[1]

    # --- Save config (master only, skipped in debug mode) ---
    legacy = config.to_legacy_dict()
    if is_master and checkpoint_dir is not None:
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(legacy, f, indent=4)

    # --- W&B (master only) ---
    if is_master:
        wandb_project = config.wandb.project + ("_sr" if config.data.super_resolution else "")
        wandb.init(
            project=wandb_project,
            entity=config.wandb.entity,
            name=run_name,
            group=config.wandb.group or None,
            config=legacy,
            mode="disabled" if config.debugging else "online",
        )

    # --- Data loaders ---
    train_loader, val_loader, traj_loader = get_data_loaders(
        config.data, is_distributed=is_distributed
    )

    # --- Build and move model ---
    model = build_model(config)
    model.to(device)

    if is_master:
        print(f"Model class  : {model_type}")
        print(f"Parameters   : {count_parameters(model):,}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # --- Loss function ---
    criterion = _make_criterion(config)

    # --- Route to the appropriate trainer ---
    prediction_steps = config.data.prediction_steps
    is_diffusion_like = isinstance(config.model, (DiffusionModelConfig, RefinerConfig))
    is_unet = isinstance(config.model, (Unet2DConfig, Unet1DConfig))

    if config.data.super_resolution and prediction_steps > 1:
        raise ValueError(
            "super_resolution=True requires prediction_steps=1. "
            "Autoregressive multi-step unrolling is not defined for super-resolution."
        )

    if is_diffusion_like:
        if prediction_steps == 1:
            train_diffusion_model(
                model,
                train_loader,
                val_loader,
                traj_loader,
                legacy["train_params"],
                criterion,
                legacy,
                checkpoint_dir,
                device=device,
                is_master=is_master,
            )
        else:
            if isinstance(config.model, RefinerConfig):
                raise NotImplementedError(
                    "Multi-step training for PDERefiner is not yet implemented. "
                    "Use prediction_steps=1 or add a dedicated refiner multistep trainer."
                )
            train_diffusion_model_multisteps(
                model,
                train_loader,
                val_loader,
                traj_loader,
                legacy["train_params"],
                criterion,
                legacy,
                checkpoint_dir,
            )

    elif is_unet:
        if prediction_steps == 1:
            train_unet(
                model,
                train_loader,
                val_loader,
                traj_loader,
                legacy["train_params"],
                criterion,
                legacy,
                checkpoint_dir,
                device=device,
                is_master=is_master,
            )
        else:
            train_unet_multisteps(
                model,
                train_loader,
                val_loader,
                traj_loader,
                legacy["train_params"],
                criterion,
                legacy,
                checkpoint_dir,
                device=device,
                is_master=is_master,
            )

    else:
        raise ValueError(f"No trainer registered for model config type '{type(config.model)}'")

    if is_master:
        wandb.finish()

    if is_distributed:
        cleanup()


if __name__ == "__main__":
    main()
