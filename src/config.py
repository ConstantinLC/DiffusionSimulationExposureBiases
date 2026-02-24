from pydantic import BaseModel
from typing import Literal, Optional
from omegaconf import DictConfig, OmegaConf


class TrainingConfig(BaseModel):
    num_epochs: int
    learning_rate_start: float = 1e-4
    learning_rate_end: float = 1e-6
    T_max: int
    device: str = "cuda"
    epoch_sampling_frequency: int = 50
    backgrad: bool = False
    n_proxy_steps: int = 0
    tau: float = 1.05
    binary_search_steps: int = 0
    finetuning_steps: int = 0
    perform_binary_search: bool = False
    best_log_sigma: Optional[float] = None
    channels_per_frame: Optional[int] = None
    first_ar_step_noising_step_limit: Optional[int] = None


class DataConfig(BaseModel):
    dataset_name: Literal["KolmogorovFlow", "KuramotoSivashinsky", "TransonicFlow"]
    data_path: str
    resolution: int
    super_resolution: bool = False
    downscale_factor: int = 4
    prediction_steps: int = 1   # replaces sequence_length[0]-1; 1=single-step, 2=multi-step
    frames_per_step: int = 1    # replaces sequence_length[1]
    traj_length: int = 64       # replaces trajectory_sequence_length[0]
    frames_per_time_step: int = 1
    limit_trajectories_train: int = -1
    limit_trajectories_val: int = -1
    batch_size: int = 64
    val_batch_size: int = 64

    @property
    def sequence_length(self) -> list[int]:
        return [self.prediction_steps + 1, self.frames_per_step]

    @property
    def trajectory_sequence_length(self) -> list[int]:
        return [self.traj_length, self.frames_per_step]


class ModelConfig(BaseModel):
    dimension: int = 2
    dataSize: list[int]
    condChannels: int
    dataChannels: int
    diffSchedule: str = "linear"
    diffSteps: int = 20
    inferenceSamplingMode: str = "ddpm"
    inferenceConditioningIntegration: str = "clean"
    diffCondIntegration: str = "clean"
    padding_mode: str = "circular"
    architecture: str = "Unet2D"
    checkpoint: str = ""
    load_betas: bool = False
    # Unet-specific fields (used when class=Unet2D or Unet1D)
    dim: Optional[int] = None
    dim_mults: list[int] = [1, 1, 1]
    convnext_mult: int = 1
    with_time_emb: bool = False


class LossConfig(BaseModel):
    name: Literal["mse", "l1"] = "mse"
    image_size: list[int]
    eval_traj_metrics: list[str] = ["mse"]
    primary_metric: str = "mse"


class WandbConfig(BaseModel):
    project: str
    entity: str = "cleclei"
    name: str = ""
    group: Optional[str] = None


class ExperimentConfig(BaseModel):
    checkpoint_dir: str
    debugging: bool = False
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    wandb: WandbConfig

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "ExperimentConfig":
        return cls(**OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    def to_legacy_dict(self) -> dict:
        """Backward-compat dict for passing to existing trainer functions."""
        return {
            "train_params": self.training.dict(),
            "data_params": {
                **self.data.dict(),
                "base_checkpoint_dir": self.checkpoint_dir,
                "sequence_length": self.data.sequence_length,
                "trajectory_sequence_length": self.data.trajectory_sequence_length,
            },
            "loss_params": self.loss.dict(),
            "model_params": self.model.dict(),
            "wandb_params": self.wandb.dict(),
        }
