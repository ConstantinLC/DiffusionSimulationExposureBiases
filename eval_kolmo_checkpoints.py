#!/usr/bin/env python
"""
Evaluate all model checkpoints under checkpoints/KolmogorovFlow.

Discovers valid model groups (DiffusionModel_*, PDERefiner, Unet2D*, …),
finds run subdirectories within each, and computes per-run metrics on the
trajectory validation dataset:
  - step1_mse       : MSE on raw field at the 1st rollout step
  - avg_mse         : average MSE on raw field over all rollout steps
  - last_step_mse   : MSE on raw field at the final rollout step
  - step1_vort_mse  : MSE on vorticity at the 1st rollout step
  - avg_vort_mse    : average MSE on vorticity over all rollout steps
  - last_vort_mse   : MSE on vorticity at the final rollout step
  - time_to_failure : mean steps until vorticity Pearson corr with GT drops below 0.8
  - fsd             : Fourier Spectral Distance at the final step
  - reb             : Relative Exposure Bias = mse_inference[0] / mse_clean[0]
                      (DiffusionModel / PDERefiner only; null for plain U-Net)
  - nfe             : number of network forward passes per prediction step

Results are saved to checkpoints/KolmogorovFlow/results.json.

Usage:
    python eval_kolmo_checkpoints.py [--device cuda] [--rollout_steps 63]
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
from src.models.diffusion import DiffusionModel
from src.models.pderefiner import PDERefiner
from src.models.unet_2d import Unet
from src.utils.general import fsd_torch_radial, vorticity

BASE = ROOT / "checkpoints" / "KolmogorovFlow"
EXPLORATION_BASE = BASE / "exploration"

# Top-level folders that contain trained models (regex patterns).
# Anything else (exploration, old++, baselines, …) is skipped.
VALID_MODEL_PATTERNS = [
    r"^DiffusionModel",
    r"^PDERefiner$",
    r"^Unet2D",
    r"^DilResNet",
    r"^FNO",
]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def is_valid_model_name(name: str) -> bool:
    return any(re.match(pat, name) for pat in VALID_MODEL_PATTERNS)


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


def discover_exploration_runs(base: Path) -> dict[str, list[Path]]:
    """Returns {run_folder_name: [greedy_trained_dir, ...]} for all run_* subfolders."""
    groups: dict[str, list[Path]] = {}
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        runs = _collect_ckpts(run_dir)
        if runs:
            groups[run_dir.name] = runs
    return groups


def discover_runs(base: Path) -> dict[str, list[Path]]:
    """
    Returns {model_group_name: [run_dir, ...]} for every valid model group.

    A run_dir is any subdirectory of a valid model group that passes
    is_valid_ckpt().  If the model group directory itself is a valid
    checkpoint it is included as well.
    """
    groups: dict[str, list[Path]] = {}

    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not is_valid_model_name(entry.name):
            continue

        runs: list[Path] = []

        # The group dir itself may be a checkpoint (e.g. Unet2D contains
        # both run_* subdirs AND its own best_model.pth).
        if is_valid_ckpt(entry):
            runs.append(entry)

        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and is_valid_ckpt(sub):
                runs.append(sub)

        if runs:
            groups[entry.name] = runs

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
    Instantiate the right model class from the legacy model_params dict,
    loading weights via the constructor's checkpoint argument where supported.
    Returns (model, arch_type) where arch_type ∈ {'diffusion', 'refiner', 'unet'}.
    """
    mp = dict(model_params)

    if "refinementSteps" in mp:
        model = PDERefiner(
            dimension=mp.get("dimension", 2),
            dataSize=mp["dataSize"],
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            refinementSteps=mp["refinementSteps"],
            log_sigma_min=mp["log_sigma_min"],
            padding_mode=mp.get("padding_mode", "circular"),
            architecture=mp.get("architecture", "Unet2D"),
            checkpoint=str(ckpt_path),
        )
        return model, "refiner"

    if "diffSchedule" in mp:
        model = DiffusionModel(
            dimension=mp.get("dimension", 2),
            dataSize=mp["dataSize"],
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            diffSchedule=mp["diffSchedule"],
            diffSteps=mp["diffSteps"],
            inferenceSamplingMode=mp["inferenceSamplingMode"],
            inferenceConditioningIntegration=mp["inferenceConditioningIntegration"],
            diffCondIntegration=mp["diffCondIntegration"],
            padding_mode=mp.get("padding_mode", "circular"),
            architecture=mp.get("architecture", "Unet2D"),
            checkpoint=str(ckpt_path),
            load_betas=False,
            schedule_path=mp.get("schedule_path"),
        )
        return model, "diffusion"

    # Unet2D (legacy JSON: has "dim", "channels", no condChannels).
    # No checkpoint argument in this class — load manually.
    model = Unet(
        dim=mp.get("dim", 64),
        channels=mp.get("channels", mp.get("condChannels", 2)),
        dim_mults=tuple(mp.get("dim_mults", [1, 1, 1])),
        convnext_mult=mp.get("convnext_mult", 1),
        with_time_emb=mp.get("with_time_emb", False),
        padding_mode=mp.get("padding_mode", "circular"),
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
    traj_len = dp.get("traj_length", dp.get("trajectory_sequence_length", [64, 1])[0])
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


def _vort_pearson_corr(vort_pred: torch.Tensor, vort_gt: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation of vorticity fields per trajectory.
    vort_pred, vort_gt: (B, H, W) → returns (B,).
    """
    B = vort_pred.shape[0]
    p = vort_pred.reshape(B, -1).float()
    g = vort_gt.reshape(B, -1).float()
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
    """Rollout metrics: step1_mse, avg_mse, last_step_mse, vorticity metrics, fsd, time_to_failure."""
    model.eval()
    all_sq_err: list[torch.Tensor] = []
    all_vort_sq_err: list[torch.Tensor] = []
    all_vort_corrs: list[torch.Tensor] = []  # each: (B, K)
    preds_last_list: list[torch.Tensor] = []
    gt_last_list: list[torch.Tensor] = []

    with torch.no_grad():
        for sample in traj_loader:
            data = sample["data"].to(device)            # (B, T, C, H, W)
            gt = data[:, 1 : rollout_steps + 1]        # (B, K, C, H, W)

            current = data[:, 0]
            step_preds = []
            for _ in range(rollout_steps):
                current = run_step(model, arch, current)
                current = torch.nan_to_num(current, nan=0.0, posinf=1e5, neginf=-1e5)
                current = torch.clamp(current, -1e5, 1e5)
                step_preds.append(current.cpu())

            preds = torch.stack(step_preds, dim=1)      # (B, K, C, H, W)
            gt_cpu = gt.cpu()
            sq_err = torch.clamp((preds - gt_cpu) ** 2, max=1e10)
            all_sq_err.append(sq_err)
            preds_last_list.append(preds[:, -1])
            gt_last_list.append(gt_cpu[:, -1])

            # Vorticity: flatten time into batch, compute, reshape back
            B, K, C, H, W = preds.shape
            vort_pred = vorticity(preds.reshape(B * K, C, H, W)).reshape(B, K, H, W)
            vort_gt   = vorticity(gt_cpu.reshape(B * K, C, H, W)).reshape(B, K, H, W)
            vort_sq_err = torch.clamp((vort_pred - vort_gt) ** 2, max=1e10)
            all_vort_sq_err.append(vort_sq_err)

            # Per-trajectory per-step vorticity Pearson correlation
            corrs = torch.stack(
                [_vort_pearson_corr(vort_pred[:, t], vort_gt[:, t]) for t in range(K)],
                dim=1,
            )  # (B, K)
            all_vort_corrs.append(corrs)

    sq_err_all = torch.cat(all_sq_err, dim=0)               # (N, K, C, H, W)
    mse_per_step = torch.mean(sq_err_all, dim=(0, 2, 3, 4)).numpy()  # (K,)

    vort_sq_err_all = torch.cat(all_vort_sq_err, dim=0)     # (N, K, H, W)
    vort_mse_per_step = torch.mean(vort_sq_err_all, dim=(0, 2, 3)).numpy()  # (K,)

    corrs_all = torch.cat(all_vort_corrs, dim=0).numpy()    # (N, K)
    N, K = corrs_all.shape
    below       = corrs_all < CORR_THRESHOLD
    first_fail  = np.argmax(below, axis=1)
    has_failed  = below.any(axis=1)
    times       = np.where(has_failed, first_fail + 1, K)
    time_to_failure = float(np.mean(times))

    preds_last = torch.cat(preds_last_list, dim=0).to(device)
    gt_last    = torch.cat(gt_last_list,    dim=0).to(device)

    try:
        fsd_val = fsd_torch_radial(preds_last, gt_last).item()
        if not np.isfinite(fsd_val):
            fsd_val = None
    except Exception:
        fsd_val = None

    def safe(v: float) -> float | None:
        return float(v) if np.isfinite(v) else None

    return {
        "step1_mse":       safe(mse_per_step[0]),
        "avg_mse":         safe(float(np.mean(mse_per_step))),
        "last_step_mse":   safe(mse_per_step[-1]),
        "step1_vort_mse":  safe(vort_mse_per_step[0]),
        "avg_vort_mse":    safe(float(np.mean(vort_mse_per_step))),
        "last_vort_mse":   safe(vort_mse_per_step[-1]),
        "time_to_failure": safe(time_to_failure),
        "fsd":             fsd_val,
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
    if arch == "unet":
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


def get_nfe(model: torch.nn.Module, arch: str) -> int:
    if arch == "diffusion":
        return int(model.timesteps)
    if arch == "refiner":
        return int(model.nTimesteps)
    return 1  # unet


def compute_group_summary(group_results: dict) -> dict:
    """Compute mean and std for each numeric metric across all successful runs."""
    metric_keys = ["step1_mse", "avg_mse", "last_step_mse",
                   "step1_vort_mse", "avg_vort_mse", "last_vort_mse",
                   "time_to_failure", "fsd", "reb", "nfe"]
    summary = {}
    for key in metric_keys:
        vals = [v[key] for v in group_results.values() if isinstance(v, dict) and v.get(key) is not None]
        if vals:
            arr = np.array(vals, dtype=float)
            summary[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return summary


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
    parser.add_argument("--rollout_steps", type=int, default=63)
    parser.add_argument("--batch_size",    type=int, default=128)
    parser.add_argument("--dry_run",       action="store_true",
                        help="Just print discovered checkpoints without evaluating")
    args = parser.parse_args()

    groups = discover_runs(BASE)

    if not groups:
        print(f"No valid model checkpoints found under {BASE}.")
        sys.exit(0)

    print(f"Discovered {len(groups)} model group(s):")
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
            run_key = run_dir.name
            # Deduplicate in case the group dir itself appears as both
            # the group entry and as a run
            if run_key in group_results:
                run_key = str(run_dir.relative_to(BASE))
            try:
                metrics = evaluate_run(run_dir, args.device, args.rollout_steps, args.batch_size)
                group_results[run_key] = metrics
                print(f"    [{run_key}] step1_mse={metrics.get('step1_mse'):.4e}  "
                      f"avg_mse={metrics.get('avg_mse'):.4e}  "
                      f"ttf={metrics.get('time_to_failure')}  "
                      f"fsd={metrics.get('fsd')}  "
                      f"reb={metrics.get('reb')}  "
                      f"nfe={metrics.get('nfe')}")
            except Exception as exc:
                print(f"    [{run_key}] FAILED: {exc}")
                group_results[run_key] = {"error": str(exc)}

        all_results[group_name] = group_results

    out_path = BASE / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # ------------------------------------------------------------------
    # exploration evaluation
    # ------------------------------------------------------------------
    if not EXPLORATION_BASE.exists():
        return

    exploration_groups = discover_exploration_runs(EXPLORATION_BASE)

    if not exploration_groups:
        print(f"No valid checkpoints found under {EXPLORATION_BASE}.")
        return

    print(f"\nDiscovered {len(exploration_groups)} exploration run(s):")
    for name, runs in exploration_groups.items():
        print(f"  {name}: {len(runs)} run(s)")
        for r in runs:
            print(f"    {r}")

    if args.dry_run:
        return

    exploration_results: dict[str, dict[str, dict]] = {}

    for run_name, run_dirs in exploration_groups.items():
        print(f"\n{'='*60}")
        print(f"  exploration run: {run_name}")
        group_results: dict[str, dict] = {}

        for run_dir in run_dirs:
            run_key = str(run_dir.relative_to(EXPLORATION_BASE))
            try:
                metrics = evaluate_run(run_dir, args.device, args.rollout_steps, args.batch_size)
                group_results[run_key] = metrics
                print(f"    [{run_dir.name}] step1_mse={metrics.get('step1_mse'):.4e}  "
                      f"avg_mse={metrics.get('avg_mse'):.4e}  "
                      f"ttf={metrics.get('time_to_failure')}  "
                      f"fsd={metrics.get('fsd')}  "
                      f"reb={metrics.get('reb')}  "
                      f"nfe={metrics.get('nfe')}")
            except Exception as exc:
                print(f"    [{run_dir.name}] FAILED: {exc}")
                group_results[run_key] = {"error": str(exc)}

        summary = compute_group_summary(group_results)
        group_results["_summary"] = summary
        print(f"  Summary for {run_name}: {summary}")
        exploration_results[run_name] = group_results

    exploration_out_path = EXPLORATION_BASE / "results.json"
    with open(exploration_out_path, "w") as f:
        json.dump(exploration_results, f, indent=2)
    print(f"\nSaved exploration results to {exploration_out_path}")


if __name__ == "__main__":
    main()
