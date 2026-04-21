from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
from omegaconf import DictConfig, OmegaConf


class TrainingConfig(BaseModel):
    num_epochs: int
    seed: Optional[int] = None
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
    end_to_end: bool = False
    track_instability: bool = False
    n_noise_samples: int = 1
    validate_every_k: int = 10
    exploration_run_dir: Optional[str] = None


class DataConfig(BaseModel):
    dataset_name: Literal["KolmogorovFlow", "KuramotoSivashinsky", "TransonicFlow", "WeatherBench"]
    variables: Optional[list[str]] = None  # WeatherBench variable keys, e.g. ['z500', 't850']
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


# ---------------------------------------------------------------------------
# Per-model config classes
# ---------------------------------------------------------------------------

class _NeuralBase(BaseModel):
    """Fields common to all model types (injected by dataset configs)."""
    dimension: int = 2
    dataSize: list[int]
    condChannels: int
    dataChannels: int
    padding_mode: str = "circular"
    checkpoint: str = ""
    architecture: str = "Unet2D"

    class Config:
        extra = "ignore"  # tolerate unknown YAML keys (e.g. nested legacy blocks)


class DiffusionModelConfig(_NeuralBase):
    type: Literal["DiffusionModel"] = "DiffusionModel"
    diffSchedule: str = "linear"
    diffSteps: int = 20
    inferenceSamplingMode: str = "ddpm"
    inferenceConditioningIntegration: str = "clean"
    diffCondIntegration: str = "clean"
    load_betas: bool = False
    schedule_path: Optional[str] = None  # path to greedy schedule JSON (used when diffSchedule="from_file")
    sigma_min: Optional[float] = None  # used by diffSchedule="log_uniform"
    sigma_max: Optional[float] = None  # used by diffSchedule="log_uniform"


class RefinerConfig(_NeuralBase):
    type: Literal["PDERefiner"] = "PDERefiner"
    refinementSteps: int = 3
    log_sigma_min: float = -1.5


class _UnetBase(_NeuralBase):
    dim: Optional[int] = None
    dim_mults: list[int] = [1, 1, 1]
    convnext_mult: int = 1
    with_time_emb: bool = False


class Unet2DConfig(_UnetBase):
    type: Literal["Unet2D"] = "Unet2D"


class Unet1DConfig(_UnetBase):
    type: Literal["Unet1D"] = "Unet1D"


class DilResNetConfig(_NeuralBase):
    type: Literal["DilResNet"] = "DilResNet"
    blocks: int = 4
    features: int = 48
    dilate: bool = True


class FNOConfig(_NeuralBase):
    type: Literal["FNO"] = "FNO"
    modes: list[int] = [16, 16]
    hidden_channels: int = 64
    n_layers: int = 4


ModelConfig = Union[DiffusionModelConfig, RefinerConfig, Unet2DConfig, Unet1DConfig, DilResNetConfig, FNOConfig]


# ---------------------------------------------------------------------------

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
    model: ModelConfig = Field(discriminator="type")
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
            "model_params": self.model.dict(exclude={"type"}),
            "wandb_params": self.wandb.dict(),
        }
