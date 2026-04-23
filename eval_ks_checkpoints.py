#!/usr/bin/env python
"""
Evaluate all model checkpoints under checkpoints/KuramotoSivashinsky.

Skips old++ and exploration; discovers everything else recursively and groups
results by top-level folder name.  Per-run metrics:
  - step1_mse       : MSE at the 1st rollout step
  - avg_mse         : average MSE over all rollout steps
  - last_step_mse   : MSE at the final rollout step
  - time_to_failure : mean steps until Pearson corr with GT drops below 0.8
  - reb             : Relative Exposure Bias = mse_inference[0] / mse_clean[0]
                      (DiffusionModel / PDERefiner only; null for plain U-Net)
  - nfe             : number of network forward passes per prediction step

Results are saved to checkpoints/KuramotoSivashinsky/results.json.

Usage:
    python eval_ks_checkpoints.py [--device cuda] [--rollout_steps 140]
                                  [--batch_size 16] [--dry_run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel, EDMDiffusionModel
from src.models.pderefiner import PDERefiner
from src.models.unet_1d import Unet1D
from src.models.unet_2d import Unet

BASE = ROOT / "checkpoints" / "KuramotoSivashinsky"
TAU_GRID_BASE = BASE / "tau_grid"

# Only top-level folders whose names match one of these patterns are evaluated.
VALID_MODEL_PATTERNS = [
    r"^DiffusionModel",
    r"^PDERefiner$",
    r"^Unet",
    r"^DilResNet",
    r"^FNO",
    #r"^exploration$",
]


def is_valid_group(name: str) -> bool:
    return any(re.match(pat, name) for pat in VALID_MODEL_PATTERNS)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def is_valid_ckpt(path: Path) -> bool:
    """True if path contains config.json + at least one weight file."""
    if not (path / "config.json").exists():
        return False
    return (path / "best_model.pth").exists() or bool(list(path.glob("epoch_*.pth")))


def _collect_ckpts(directory: Path, max_depth: int = 4) -> list[Path]:
    """Recursively collect valid checkpoint directories up to max_depth."""
    if max_depth == 0:
        return []
    found = []
    for sub in sorted(directory.iterdir()):
        if not sub.is_dir():
            continue
        if is_valid_ckpt(sub):
            found.append(sub)
        else:
            found.extend(_collect_ckpts(sub, max_depth - 1))
    return found


def discover_runs(base: Path) -> dict[str, list[Path]]:
    """
    Returns {group_name: [run_dir, ...]} for every non-skipped top-level folder.
    """
    groups: dict[str, list[Path]] = {}

    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not is_valid_group(entry.name):
            continue

        runs = _collect_ckpts(entry)
        if runs:
            groups[entry.name] = runs

    return groups


def discover_tau_runs(base: Path) -> dict[str, list[Path]]:
    """Returns {tau_folder_name: [greedy_trained_dir, ...]} for all tau_* subfolders."""
    groups: dict[str, list[Path]] = {}
    for tau_dir in sorted(base.iterdir()):
        if not tau_dir.is_dir() or not tau_dir.name.startswith("tau_"):
            continue
        runs = _collect_ckpts(tau_dir)
        if runs:
            groups[tau_dir.name] = runs
    return groups


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _resolve_ckpt_path(ckpt_dir: Path) -> Path:
    ckpt_path = ckpt_dir / "best_model.pth"
    if not ckpt_path.exists():
        epoch_ckpts = sorted(
            ckpt_dir.glob("epoch_*.pth"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        ckpt_path = epoch_ckpts[-1]
        print(f"    best_model.pth not found; using {ckpt_path.name}")
    return ckpt_path


def build_model(model_params: dict, ckpt_path: Path) -> tuple[torch.nn.Module, str]:
    """
    Instantiate the right model class and load weights.
    Returns (model, arch_type) where arch_type ∈ {'diffusion', 'refiner', 'unet'}.
    """
    mp = dict(model_params)

    if "refinementSteps" in mp:
        model = PDERefiner(
            dimension=mp.get("dimension", 1),
            dataSize=mp["dataSize"],
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            refinementSteps=mp["refinementSteps"],
            log_sigma_min=mp["log_sigma_min"],
            padding_mode=mp.get("padding_mode", "circular"),
            architecture=mp.get("architecture", "Unet1D"),
            checkpoint=str(ckpt_path),
        )
        return model, "refiner"

    if "diffSchedule" in mp:
        model = DiffusionModel(
            dimension=mp.get("dimension", 1),
            dataSize=mp["dataSize"],
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            diffSchedule=mp["diffSchedule"],
            diffSteps=mp["diffSteps"],
            inferenceSamplingMode=mp["inferenceSamplingMode"],
            inferenceConditioningIntegration=mp["inferenceConditioningIntegration"],
            diffCondIntegration=mp["diffCondIntegration"],
            padding_mode=mp.get("padding_mode", "circular"),
            architecture=mp.get("architecture", "Unet1D"),
            checkpoint=str(ckpt_path),
            load_betas=False,
            schedule_path=mp.get("schedule_path"),
        )
        return model, "diffusion"

    if "num_steps" in mp and "sigma_min" in mp and "sigma_max" in mp:
        model = EDMDiffusionModel(
            dimension=mp.get("dimension", 1),
            dataSize=mp["dataSize"],
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            num_steps=mp["num_steps"],
            sigma_min=mp["sigma_min"],
            sigma_max=mp["sigma_max"],
            sigma_data=mp.get("sigma_data", 0.5),
            P_mean=mp.get("P_mean", -1.2),
            P_std=mp.get("P_std", 1.2),
            rho=mp.get("rho", 7.0),
            solver=mp.get("solver", "heun"),
            stochastic=mp.get("stochastic", False),
            S_churn=mp.get("S_churn", 10.0),
            S_tmin=mp.get("S_tmin", 0.0),
            S_tmax=mp.get("S_tmax", 1e6),
            S_noise=mp.get("S_noise", 1.0),
            padding_mode=mp.get("padding_mode", "circular"),
            architecture=mp.get("architecture", "Unet1D"),
            checkpoint=str(ckpt_path),
        )
        return model, "edm"

    # Unet1D / Unet2D (legacy JSON: has "dim", "channels", no condChannels).
    # Load weights manually since there is no checkpoint argument.
    dim = mp.get("dim", 64)
    channels = mp.get("channels", mp.get("condChannels", 1))
    dim_mults = tuple(mp.get("dim_mults", [1, 1, 1]))
    convnext_mult = mp.get("convnext_mult", 1)
    with_time_emb = mp.get("with_time_emb", False)
    padding_mode = mp.get("padding_mode", "circular")

    dimension = mp.get("dimension", 1)
    if dimension == 1:
        model = Unet1D(
            dim=dim,
            channels=channels,
            dim_mults=dim_mults,
            convnext_mult=convnext_mult,
            with_time_emb=with_time_emb,
            padding_mode=padding_mode,
        )
    else:
        model = Unet(
            dim=dim,
            channels=channels,
            dim_mults=dim_mults,
            convnext_mult=convnext_mult,
            with_time_emb=with_time_emb,
            padding_mode=padding_mode,
        )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model, "unet"


def load_model(ckpt_dir: Path, device: str) -> tuple[torch.nn.Module, str, dict]:
    """Load model weights and return (model, arch_type, full_config)."""
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)

    ckpt_path = _resolve_ckpt_path(ckpt_dir)
    model, arch = build_model(cfg["model_params"], ckpt_path)
    model.to(device).eval()
    print(f"    Loaded {arch} from {ckpt_path}")
    return model, arch, cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_data_loader(cfg: dict, batch_size: int, rollout_steps: int):
    """Build the trajectory validation loader from a run's config.json."""
    dp = cfg["data_params"]
    traj_len = dp.get("traj_length", dp.get("trajectory_sequence_length", [141, 1])[0])
    traj_len = max(traj_len, rollout_steps + 1)

    data_config = DataConfig(
        dataset_name=dp["dataset_name"],
        data_path=dp["data_path"],
        resolution=dp["resolution"],
        super_resolution=dp.get("super_resolution", False),
        downscale_factor=dp.get("downscale_factor", 4),
        prediction_steps=dp.get("prediction_steps", dp.get("sequence_length", [2, 1])[0] - 1),
        frames_per_step=dp.get("frames_per_step", dp.get("sequence_length", [2, 1])[1]),
        traj_length=traj_len,
        frames_per_time_step=dp.get("frames_per_time_step", 1),
        limit_trajectories_train=dp.get("limit_trajectories_train", -1),
        limit_trajectories_val=dp.get("limit_trajectories_val", -1),
        batch_size=batch_size,
        val_batch_size=batch_size,
    )
    _, _, traj_loader = get_data_loaders(data_config)
    return traj_loader


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_step(model: torch.nn.Module, arch: str, conditioning: torch.Tensor) -> torch.Tensor:
    if arch == "unet":
        return model(conditioning, time=None)
    return model(conditioning)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

CORR_THRESHOLD = 0.8


def _pearson_corr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation per trajectory, flattening all non-batch dims.
    pred, gt: (B, ...) → returns (B,).
    """
    B = pred.shape[0]
    p = pred.reshape(B, -1).float()
    g = gt.reshape(B, -1).float()
    p = p - p.mean(dim=1, keepdim=True)
    g = g - g.mean(dim=1, keepdim=True)
    num   = (p * g).sum(dim=1)
    denom = torch.sqrt((p ** 2).sum(dim=1) * (g ** 2).sum(dim=1))
    corr  = num / (denom + 1e-8)
    corr[denom < 1e-8] = 0.0
    return corr  # (B,)


def compute_trajectory_metrics(
    model: torch.nn.Module,
    arch: str,
    traj_loader,
    device: str,
    rollout_steps: int,
) -> dict:
    """Rollout metrics: step1_mse, avg_mse, last_step_mse, time_to_failure."""
    model.eval()
    all_sq_err:  list[torch.Tensor] = []
    all_corrs:   list[torch.Tensor] = []  # each entry: (B, K)

    with torch.no_grad():
        for sample in traj_loader:
            data = sample["data"].to(device)        # (B, T, C, L)
            gt = data[:, 1 : rollout_steps + 1]    # (B, K, C, L)

            current = data[:, 0]
            step_preds = []
            for _ in range(rollout_steps):
                current = run_step(model, arch, current)
                current = torch.nan_to_num(current, nan=0.0, posinf=1e5, neginf=-1e5)
                current = torch.clamp(current, -1e5, 1e5)
                step_preds.append(current.cpu())

            preds  = torch.stack(step_preds, dim=1)  # (B, K, C, L)
            gt_cpu = gt.cpu()

            sq_err = torch.clamp((preds - gt_cpu) ** 2, max=1e10)
            all_sq_err.append(sq_err)

            # Per-trajectory per-step Pearson correlation
            B, K = preds.shape[:2]
            corrs = torch.stack(
                [_pearson_corr(preds[:, t], gt_cpu[:, t]) for t in range(K)],
                dim=1,
            )  # (B, K)
            all_corrs.append(corrs)

    sq_err_all = torch.cat(all_sq_err, dim=0)                    # (N, K, C, L)
    mse_per_step = torch.mean(sq_err_all, dim=(0, 2, 3)).numpy() # (K,)

    corrs_all = torch.cat(all_corrs, dim=0).numpy()  # (N, K)
    N, K = corrs_all.shape

    below = corrs_all < CORR_THRESHOLD                      # (N, K)
    first_fail = np.argmax(below, axis=1)                   # (N,) — index of first failure
    has_failed  = below.any(axis=1)                         # (N,)
    times = np.where(has_failed, first_fail + 1, K)         # steps until failure (1-indexed)
    time_to_failure = float(np.mean(times))

    def safe(v: float) -> float | None:
        return float(v) if np.isfinite(v) else None

    return {
        "step1_mse":       safe(mse_per_step[0]),
        "avg_mse":         safe(float(np.mean(mse_per_step))),
        "last_step_mse":   safe(mse_per_step[-1]),
        "time_to_failure": safe(time_to_failure),
    }


def compute_reb(
    model: torch.nn.Module,
    arch: str,
    traj_loader,
    device: str,
) -> float | None:
    """
    Relative Exposure Bias = mean(MSE_own_pred) / mean(MSE_clean).
    Only defined for DiffusionModel and PDERefiner; returns None for Unet.
    """
    if arch in ("unet", "edm"):
        return None

    model.eval()
    mse_clean_list: list[float] = []
    mse_own_list:   list[float] = []

    with torch.no_grad():
        for sample in traj_loader:
            data = sample["data"].to(device)
            cond   = data[:, 0]
            target = data[:, 1]

            pred_clean, _ = model(
                conditioning=cond, data=target,
                return_x0_estimate=True, input_type="clean",
            )
            pred_own, _ = model(
                conditioning=cond, data=target,
                return_x0_estimate=True, input_type="own-pred",
            )

            mse_clean_list.append(torch.mean((pred_clean - target) ** 2).item())
            mse_own_list.append(  torch.mean((pred_own   - target) ** 2).item())

    mse_clean = float(np.mean(mse_clean_list))
    mse_own   = float(np.mean(mse_own_list))

    if mse_clean < 1e-15:
        return None
    reb = mse_own / mse_clean
    return float(reb) if np.isfinite(reb) else None


def compute_group_summary(group_results: dict) -> dict:
    """Compute mean and std for each numeric metric across all successful runs."""
    metric_keys = ["step1_mse", "avg_mse", "last_step_mse", "time_to_failure", "reb", "nfe"]
    summary = {}
    for key in metric_keys:
        vals = [v[key] for v in group_results.values() if isinstance(v, dict) and v.get(key) is not None]
        if vals:
            arr = np.array(vals, dtype=float)
            summary[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return summary


def get_nfe(model: torch.nn.Module, arch: str) -> int:
    if arch == "diffusion":
        return int(model.timesteps)
    if arch == "refiner":
        return int(model.nTimesteps)
    if arch == "edm":
        return int(model.num_steps)
    return 1


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    device: str,
    rollout_steps: int,
    batch_size: int,
) -> dict:
    print(f"  Evaluating {run_dir}")
    model, arch, cfg = load_model(run_dir, device)
    traj_loader = make_data_loader(cfg, batch_size, rollout_steps)

    traj_metrics = compute_trajectory_metrics(model, arch, traj_loader, device, rollout_steps)
    reb          = compute_reb(model, arch, traj_loader, device)
    nfe          = get_nfe(model, arch)

    return {
        "run_dir":  str(run_dir),
        "arch":     arch,
        "nfe":      nfe,
        **traj_metrics,
        "reb":      reb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--rollout_steps", type=int, default=140)
    parser.add_argument("--batch_size",    type=int, default=128)
    parser.add_argument("--dry_run",       action="store_true",
                        help="Print discovered checkpoints without evaluating")
    args = parser.parse_args()

    groups = discover_runs(BASE)

    if not groups:
        print(f"No valid model checkpoints found under {BASE}.")
        sys.exit(0)

    print(f"Discovered {len(groups)} group(s):")
    for name, runs in groups.items():
        print(f"  {name}: {len(runs)} run(s)")
        for r in runs:
            print(f"    {r}")

    if args.dry_run:
        return

    all_results: dict[str, dict[str, dict]] = {}

    for group_name, run_dirs in groups.items():
        print(f"\n{'='*60}")
        print(f"  Group: {group_name}")
        group_results: dict[str, dict] = {}

        for run_dir in run_dirs:
            run_key = str(run_dir.relative_to(BASE))
            try:
                metrics = evaluate_run(run_dir, args.device, args.rollout_steps, args.batch_size)
                group_results[run_key] = metrics
                print(f"    [{run_dir.name}] step1_mse={metrics.get('step1_mse'):.4e}  "
                      f"avg_mse={metrics.get('avg_mse'):.4e}  "
                      f"ttf={metrics.get('time_to_failure')}  "
                      f"reb={metrics.get('reb')}  "
                      f"nfe={metrics.get('nfe')}")
            except Exception as exc:
                print(f"    [{run_dir.name}] FAILED: {exc}")
                group_results[run_key] = {"error": str(exc)}

        summary = compute_group_summary(group_results)
        group_results["_summary"] = summary
        print(f"  Summary for {group_name}: {summary}")
        all_results[group_name] = group_results

    out_path = BASE / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # ------------------------------------------------------------------
    # tau_grid evaluation
    # ------------------------------------------------------------------
    tau_groups = discover_tau_runs(TAU_GRID_BASE)

    if not tau_groups:
        print(f"No valid checkpoints found under {TAU_GRID_BASE}.")
        return

    print(f"\nDiscovered {len(tau_groups)} tau group(s):")
    for name, runs in tau_groups.items():
        print(f"  {name}: {len(runs)} run(s)")
        for r in runs:
            print(f"    {r}")

    if args.dry_run:
        return

    tau_results: dict[str, dict[str, dict]] = {}

    for tau_name, run_dirs in tau_groups.items():
        print(f"\n{'='*60}")
        print(f"  tau group: {tau_name}")
        group_results = {}

        for run_dir in run_dirs:
            run_key = str(run_dir.relative_to(TAU_GRID_BASE))
            try:
                metrics = evaluate_run(run_dir, args.device, args.rollout_steps, args.batch_size)
                group_results[run_key] = metrics
                print(f"    [{run_dir.name}] step1_mse={metrics.get('step1_mse'):.4e}  "
                      f"avg_mse={metrics.get('avg_mse'):.4e}  "
                      f"ttf={metrics.get('time_to_failure')}  "
                      f"reb={metrics.get('reb')}  "
                      f"nfe={metrics.get('nfe')}")
            except Exception as exc:
                print(f"    [{run_dir.name}] FAILED: {exc}")
                group_results[run_key] = {"error": str(exc)}

        summary = compute_group_summary(group_results)
        group_results["_summary"] = summary
        print(f"  Summary for {tau_name}: {summary}")
        tau_results[tau_name] = group_results

    tau_out_path = TAU_GRID_BASE / "results.json"
    with open(tau_out_path, "w") as f:
        json.dump(tau_results, f, indent=2)
    print(f"\nSaved tau_grid results to {tau_out_path}")


if __name__ == "__main__":
    main()
