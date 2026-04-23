#!/usr/bin/env python
"""
Evaluate all model checkpoints under checkpoints/WeatherBench.

Discovers valid model groups (DiffusionModel_*, PDERefiner, Unet2D*, …),
finds run subdirectories within each, and computes per-run metrics on the
trajectory validation dataset.  The WeatherBench data is 6-hourly, so:

  - 6 h  = rollout step  1  (index 0 in 0-based arrays)
  - 3 d  = rollout step 12  (index 11)
  - 5 d  = rollout step 20  (index 19)

Per-run metrics:
  - rmse_6h / rmse_3d / rmse_5d  : latitude-weighted RMSE in physical units
                                    per variable, e.g. rmse_6h_z500
  - acc_6h  / acc_3d  / acc_5d   : latitude-weighted spatial ACC per variable
                                    (anomaly = normalised field, ~0-mean by construction)
  - fsd                          : Fourier Spectral Distance at the final step
  - nfe                          : network forward passes per prediction step

Results are saved to checkpoints/WeatherBench/results.json.

Usage:
    python eval_wb_checkpoints.py [--device cuda] [--rollout_steps 20]
                                  [--batch_size 32] [--dry_run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import WeatherBenchDataset
from src.models.diffusion import DiffusionModel
from src.models.pderefiner import PDERefiner
from src.models.unet_2d import Unet
from src.utils.general import fsd_torch_radial

BASE = ROOT / "checkpoints" / "WeatherBench"

VALID_MODEL_PATTERNS = [
    r"^DiffusionModel",
    r"^PDERefiner",
    r"^Unet2D",
    #r"^DilResNet",
    #r"^FNO",
]

# WeatherBench 2.8125° grid: 64 latitudes from -88.59375° to +88.59375°
_LATS = np.linspace(-88.59375, 88.59375, 64).astype(np.float32)
_LAT_W = np.cos(np.deg2rad(_LATS))
_LAT_W /= _LAT_W.mean()   # renormalise so that flat weighting ≡ mean weight=1

# Step indices (0-based) for the three evaluation horizons
EVAL_STEPS = {"6h": 0, "3d": 11, "5d": 19}

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def is_valid_model_name(name: str) -> bool:
    return any(re.match(pat, name) for pat in VALID_MODEL_PATTERNS)


def is_valid_ckpt(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return (path / "best_model.pth").exists() or bool(list(path.glob("epoch_*.pth")))


def _collect_ckpts(directory: Path, max_depth: int = 4) -> list[Path]:
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
    groups: dict[str, list[Path]] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not is_valid_model_name(entry.name):
            continue
        runs = _collect_ckpts(entry)
        if is_valid_ckpt(entry):
            runs.insert(0, entry)
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
    mp = dict(model_params)

    if "refinementSteps" in mp:
        model = PDERefiner(
            dimension=mp.get("dimension", 2),
            dataSize=mp["dataSize"],
            condChannels=mp["condChannels"],
            dataChannels=mp["dataChannels"],
            refinementSteps=mp["refinementSteps"],
            log_sigma_min=mp["log_sigma_min"],
            padding_mode=mp.get("padding_mode", "lonlat"),
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
            padding_mode=mp.get("padding_mode", "lonlat"),
            architecture=mp.get("architecture", "Unet2D"),
            checkpoint=str(ckpt_path),
            load_betas=False,
        )
        return model, "diffusion"

    model = Unet(
        dim=mp.get("dim", 64),
        channels=mp.get("channels", mp.get("condChannels", 2)),
        dim_mults=tuple(mp.get("dim_mults", [1, 1, 1])),
        convnext_mult=mp.get("convnext_mult", 1),
        with_time_emb=mp.get("with_time_emb", False),
        padding_mode=mp.get("padding_mode", "lonlat"),
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model, "unet"


def load_model(ckpt_dir: Path, device: str) -> tuple[torch.nn.Module, str, dict]:
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
    """
    Returns (traj_loader, variables) for the WeatherBench validation set.
    Stats are read from {data_path}/.wb_norm_stats.json (auto-populated on first use).
    """
    dp = cfg["data_params"]
    variables: list[str] = dp.get("variables", ["z500", "t850"])
    frames_per_step: int = dp.get("frames_per_step", dp.get("sequence_length", [2, 1])[1])

    traj_len = dp.get("traj_length", dp.get("trajectory_sequence_length", [56, 1])[0])
    traj_len = max(traj_len, rollout_steps + 1)

    # WeatherBenchDataset reads stats from cache or computes them from the
    # training split (one-time cost) and writes them to the cache automatically.
    train_set = WeatherBenchDataset(
        "WeatherBench-train",
        dp["data_path"],
        mode="train",
        variables=variables,
        sequenceLength=[2, frames_per_step],
    )
    traj_set = WeatherBenchDataset(
        "WeatherBench-traj",
        dp["data_path"],
        mode="valid",
        variables=variables,
        sequenceLength=[traj_len, frames_per_step],
        mean=train_set.mean,
        std=train_set.std,
        stride=rollout_steps,
    )
    traj_loader = DataLoader(
        traj_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    # (1, C, 1, 1) tensors for unnormalisation inside compute_trajectory_metrics
    mean = torch.from_numpy(train_set.mean)  # float32
    std  = torch.from_numpy(train_set.std)
    return traj_loader, variables, mean, std


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_step(model: torch.nn.Module, arch: str, conditioning: torch.Tensor) -> torch.Tensor:
    if arch == "unet":
        return model(conditioning, time=None)
    return model(conditioning)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _lat_weight(device: str) -> torch.Tensor:
    return torch.from_numpy(_LAT_W).to(device).reshape(1, -1, 1)   # (1, H, 1)


def _weighted_mse(sq_err: torch.Tensor, lat_w: torch.Tensor) -> torch.Tensor:
    """sq_err: (B, H, W); lat_w: (1, H, 1). Returns scalar MSE."""
    return (sq_err * lat_w).mean()


def _spatial_acc(pred: torch.Tensor, gt: torch.Tensor, lat_w: torch.Tensor) -> float:
    """
    Latitude-weighted spatial ACC, averaged over batch.
    pred, gt: (B, H, W).  lat_w: (1, H, 1).
    Treats normalised field as anomaly (global mean removed by normalisation).
    """
    num   = (pred * gt * lat_w).sum(dim=(-2, -1))                           # (B,)
    denom = torch.sqrt(
        (pred ** 2 * lat_w).sum(dim=(-2, -1)) *
        (gt   ** 2 * lat_w).sum(dim=(-2, -1))
    )                                                                         # (B,)
    acc = (num / (denom + 1e-8)).mean()
    return float(acc.cpu())


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(
    model: torch.nn.Module,
    arch: str,
    traj_loader,
    variables: list[str],
    device: str,
    rollout_steps: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict:
    """
    Returns per-horizon RMSE (physical units) and ACC per variable, plus FSD.
    mean/std: (1, C, 1, 1) tensors used to unnormalise before RMSE computation.
    ACC is scale-invariant so it is computed in normalised space.
    """
    model.eval()
    lat_w = _lat_weight(device)
    # per-variable std scalars for unnormalising: shape (C,)
    std_per_var = std.reshape(-1)   # (C,)

    # Accumulators: {horizon: {var: [mse_values]}}
    rmse_acc: dict[str, dict[str, list]] = {h: {v: [] for v in variables} for h in EVAL_STEPS}
    acc_acc:  dict[str, dict[str, list]] = {h: {v: [] for v in variables} for h in EVAL_STEPS}

    preds_last_list: list[torch.Tensor] = []
    gt_last_list:    list[torch.Tensor] = []

    with torch.no_grad():
        for sample in traj_loader:
            data = sample["data"].to(device)            # (B, T, C, H, W)
            gt   = data[:, 1 : rollout_steps + 1]      # (B, K, C, H, W)

            current = data[:, 0]
            step_preds = []
            for _ in range(rollout_steps):
                current = run_step(model, arch, current)
                current = torch.nan_to_num(current, nan=0.0, posinf=1e4, neginf=-1e4)
                current = torch.clamp(current, -1e4, 1e4)
                step_preds.append(current.cpu())

            preds  = torch.stack(step_preds, dim=1)    # (B, K, C, H, W)
            gt_cpu = gt.cpu()

            for horizon, idx in EVAL_STEPS.items():
                if idx >= rollout_steps:
                    continue
                for c, var in enumerate(variables):
                    p = preds[:, idx, c]    # (B, H, W) — normalised
                    g = gt_cpu[:, idx, c]   # (B, H, W) — normalised
                    # RMSE in physical units: unnormalise by multiplying with per-var std
                    s = std_per_var[c].item()
                    sq_err = torch.clamp(((p - g) * s) ** 2, max=1e12)
                    rmse_acc[horizon][var].append(
                        _weighted_mse(sq_err, lat_w.cpu()).item()
                    )
                    # ACC is scale-invariant: compute in normalised space
                    acc_acc[horizon][var].append(_spatial_acc(p, g, lat_w.cpu()))

            preds_last_list.append(preds[:, -1])    # (B, C, H, W)
            gt_last_list.append(gt_cpu[:, -1])

    # Aggregate
    def safe(v: float) -> float | None:
        return float(v) if np.isfinite(v) else None

    metrics: dict = {}
    for horizon in EVAL_STEPS:
        for var in variables:
            mse_vals = rmse_acc[horizon][var]
            acc_vals = acc_acc[horizon][var]
            if mse_vals:
                metrics[f"rmse_{horizon}_{var}"] = safe(np.sqrt(np.mean(mse_vals)))
                metrics[f"acc_{horizon}_{var}"]  = safe(np.mean(acc_vals))
            else:
                metrics[f"rmse_{horizon}_{var}"] = None
                metrics[f"acc_{horizon}_{var}"]  = None

    # FSD at final rollout step
    preds_last = torch.cat(preds_last_list, dim=0).to(device)   # (N, C, H, W)
    gt_last    = torch.cat(gt_last_list,    dim=0).to(device)
    try:
        fsd_val = fsd_torch_radial(preds_last, gt_last).item()
        metrics["fsd"] = fsd_val if np.isfinite(fsd_val) else None
    except Exception:
        metrics["fsd"] = None

    return metrics


def get_nfe(model: torch.nn.Module, arch: str) -> int:
    if arch == "diffusion":
        return int(model.timesteps)
    if arch == "refiner":
        return int(model.nTimesteps)
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
    traj_loader, variables, mean, std = make_data_loader(cfg, batch_size, rollout_steps)

    traj_metrics = compute_trajectory_metrics(
        model, arch, traj_loader, variables, device, rollout_steps, mean, std,
    )
    nfe = get_nfe(model, arch)

    return {
        "run_dir":   str(run_dir),
        "arch":      arch,
        "variables": variables,
        "nfe":       nfe,
        **traj_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--rollout_steps", type=int, default=20,
                        help="Must be ≥20 to cover 5-day horizon")
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--dry_run",       action="store_true",
                        help="Print discovered checkpoints without evaluating")
    args = parser.parse_args()

    if args.rollout_steps < max(EVAL_STEPS.values()) + 1:
        print(f"WARNING: rollout_steps={args.rollout_steps} is too short for "
              f"all horizons; setting to {max(EVAL_STEPS.values()) + 1}")
        args.rollout_steps = max(EVAL_STEPS.values()) + 1

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
            run_key = str(run_dir.relative_to(BASE))
            try:
                metrics = evaluate_run(run_dir, args.device, args.rollout_steps, args.batch_size)
                group_results[run_key] = metrics

                vars_ = metrics.get("variables", ["z500", "t850"])
                rmse_parts = "  ".join(
                    f"rmse_5d_{v}={metrics.get(f'rmse_5d_{v}'):.4f}"
                    for v in vars_ if metrics.get(f"rmse_5d_{v}") is not None
                )
                acc_parts = "  ".join(
                    f"acc_5d_{v}={metrics.get(f'acc_5d_{v}'):.4f}"
                    for v in vars_ if metrics.get(f"acc_5d_{v}") is not None
                )
                print(f"    [{run_dir.name}]  {rmse_parts}  {acc_parts}  "
                      f"fsd={metrics.get('fsd')}  nfe={metrics.get('nfe')}")
            except Exception as exc:
                import traceback
                print(f"    [{run_dir.name}] FAILED: {exc}")
                traceback.print_exc()
                group_results[run_key] = {"error": str(exc)}

        all_results[group_name] = group_results

    out_path = BASE / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
