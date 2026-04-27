#!/usr/bin/env python
"""
Evaluate the last model checkpoint from an online algorithm run.

Loads online_state.json, reconstructs the model with the discovered final schedule,
and computes:
  - Trajectory rollout metrics (step1_mse, avg_mse, last_step_mse, time_to_failure)
  - Per-level clean MSE and inference MSE over the final schedule

Results are printed and saved to <run_dir>/eval_results.json.

Usage:
    python eval_online_run.py [--run_dir PATH] [--device cuda] [--rollout_steps 140]
                              [--batch_size 64] [--n_eval_batches 30]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel

DEFAULT_RUN_DIR = ROOT / "checkpoints/KuramotoSivashinsky/online/run_15"

KS_DATA_PARAMS = dict(
    dataset_name="KuramotoSivashinsky",
    data_path="/mnt/SSD2/constantin/archives/LPSDA/data_og",
    resolution=64,
    frames_per_step=4,
    traj_length=160,
    frames_per_time_step=1,
    limit_trajectories_train=-1,
    limit_trajectories_val=64,
)

CORR_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_and_load_model(ckpt_path: Path, final_schedule_lohi: torch.Tensor, device: str) -> DiffusionModel:
    """Build a KS DiffusionModel and load weights from an online checkpoint."""
    model = DiffusionModel(
        dimension=1,
        dataSize=[64],
        condChannels=1,
        dataChannels=1,
        diffSchedule="linear",
        diffSteps=40,  # placeholder; overridden by compute_schedule_variables below
        inferenceSamplingMode="ddpm",
        inferenceConditioningIntegration="clean",
        diffCondIntegration="clean",
        padding_mode="circular",
        checkpoint="",
        load_betas=False,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Exclude 'sigmas' — it's schedule-dependent and reset by compute_schedule_variables
    unet_state = {k[5:]: v for k, v in state.items() if k.startswith("unet.") and k != "unet.sigmas"}
    model.unet.load_state_dict(unet_state, strict=False)

    model.compute_schedule_variables(final_schedule_lohi)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _make_data_config(batch_size: int, traj_length: int) -> DataConfig:
    params = {**KS_DATA_PARAMS, "traj_length": traj_length}
    return DataConfig(
        **params,
        prediction_steps=1,
        super_resolution=False,
        downscale_factor=4,
        batch_size=batch_size,
        val_batch_size=batch_size,
    )


def make_traj_loader(batch_size: int, rollout_steps: int):
    traj_len = max(KS_DATA_PARAMS["traj_length"], rollout_steps + 1)
    _, _, traj_loader = get_data_loaders(_make_data_config(batch_size, traj_len))
    return traj_loader


def make_val_loader(batch_size: int):
    _, val_loader, _ = get_data_loaders(_make_data_config(batch_size, KS_DATA_PARAMS["traj_length"]))
    return val_loader


# ---------------------------------------------------------------------------
# Trajectory rollout metrics
# ---------------------------------------------------------------------------

def _pearson_corr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    p = pred.reshape(B, -1).float()
    g = gt.reshape(B, -1).float()
    p = p - p.mean(dim=1, keepdim=True)
    g = g - g.mean(dim=1, keepdim=True)
    num   = (p * g).sum(dim=1)
    denom = torch.sqrt((p ** 2).sum(dim=1) * (g ** 2).sum(dim=1))
    corr  = num / (denom + 1e-8)
    corr[denom < 1e-8] = 0.0
    return corr


def compute_trajectory_metrics(model: DiffusionModel, traj_loader, device: str, rollout_steps: int) -> dict:
    model.eval()
    all_sq_err, all_corrs = [], []

    with torch.no_grad():
        for sample in traj_loader:
            data = sample["data"].to(device)
            gt   = data[:, 1 : rollout_steps + 1]

            current    = data[:, 0]
            step_preds = []
            for _ in range(rollout_steps):
                current = model(current)
                current = torch.nan_to_num(current, nan=0.0, posinf=1e5, neginf=-1e5).clamp(-1e5, 1e5)
                step_preds.append(current.cpu())

            preds  = torch.stack(step_preds, dim=1)
            gt_cpu = gt.cpu()

            sq_err = (preds - gt_cpu) ** 2
            all_sq_err.append(sq_err)

            B, K = preds.shape[:2]
            corrs = torch.stack(
                [_pearson_corr(preds[:, t], gt_cpu[:, t]) for t in range(K)], dim=1
            )
            all_corrs.append(corrs)

    sq_err_all   = torch.cat(all_sq_err, dim=0)
    mse_per_step = torch.mean(sq_err_all, dim=(0, 2, 3)).numpy()

    corrs_all  = torch.cat(all_corrs, dim=0).numpy()
    below      = corrs_all < CORR_THRESHOLD
    first_fail = np.argmax(below, axis=1)
    has_failed = below.any(axis=1)
    K          = corrs_all.shape[1]
    times      = np.where(has_failed, first_fail + 1, K)

    def safe(v):
        return float(v) if np.isfinite(v) else None

    return {
        "step1_mse":       safe(mse_per_step[0]),
        "avg_mse":         safe(float(np.mean(mse_per_step))),
        "last_step_mse":   safe(mse_per_step[-1]),
        "time_to_failure": safe(float(np.mean(times))),
        "mse_per_step":    [safe(v) for v in mse_per_step.tolist()],
    }


# ---------------------------------------------------------------------------
# Per-level clean and inference errors
# ---------------------------------------------------------------------------

def compute_per_level_errors(
    model: DiffusionModel,
    val_loader,
    device: str,
    final_schedule_lohi: torch.Tensor,
    n_batches: int = 30,
) -> tuple[list, list, list]:
    """
    For each sigma in final_schedule_lohi compute clean-input MSE and
    inference-chain MSE, then restore the schedule.
    """
    model.eval()
    model.compute_schedule_variables(final_schedule_lohi.to(device))

    n_steps = final_schedule_lohi.shape[0]
    sqrtA   = model.sqrtAlphasCumprod
    sqrtOMA = model.sqrtOneMinusAlphasCumprod
    C_cond  = model.condChannels

    all_clean_mse = [[] for _ in range(n_steps)]
    all_inf_mse   = [[] for _ in range(n_steps)]
    spatial_dims  = None

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            if batch_idx >= n_batches:
                break

            data   = sample["data"].to(device)
            cond   = data[:, 0]
            target = data[:, 1]
            B      = target.shape[0]

            if spatial_dims is None:
                spatial_dims = tuple(range(1, target.ndim))

            for t_idx in range(n_steps):
                t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)
                eps   = torch.randn_like(target)
                y     = sqrtA[t_vec] * target + sqrtOMA[t_vec] * eps
                inp   = torch.cat((cond, y), dim=1)
                pred  = model.unet(inp, t_vec)[:, C_cond:]
                x0    = (y - sqrtOMA[t_vec] * pred) / sqrtA[t_vec].clamp(min=1e-8)
                all_clean_mse[t_idx].append((x0 - target).pow(2).mean(dim=spatial_dims).cpu())

            x = torch.randn_like(target)
            for step in range(n_steps - 1, -1, -1):
                t_cur  = torch.full((B,), step, device=device, dtype=torch.long)
                inp    = torch.cat((cond, x), dim=1)
                pred   = model.unet(inp, t_cur)[:, C_cond:]
                x0_hat = (x - sqrtOMA[t_cur] * pred) / sqrtA[t_cur].clamp(min=1e-8)
                all_inf_mse[step].append((x0_hat - target).pow(2).mean(dim=spatial_dims).cpu())
                if step > 0:
                    t_next = torch.full((B,), step - 1, device=device, dtype=torch.long)
                    x = sqrtA[t_next] * x0_hat + sqrtOMA[t_next] * torch.randn_like(target)

    sigmas     = final_schedule_lohi.tolist()
    clean_mses = [torch.cat(all_clean_mse[t]).mean().item() if all_clean_mse[t] else float("nan") for t in range(n_steps)]
    inf_mses   = [torch.cat(all_inf_mse[t]).mean().item()   if all_inf_mse[t]   else float("nan") for t in range(n_steps)]
    return sigmas, clean_mses, inf_mses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",       default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--rollout_steps", type=int, default=140)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--n_eval_batches",type=int, default=30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    state_path = run_dir / "online_state.json"
    if not state_path.exists():
        print(f"ERROR: {state_path} not found")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    # Final schedule is stored high→low; convert to low→high for the model
    final_schedule_lohi = torch.tensor(sorted(state["final_schedule"]), dtype=torch.float32)
    n_steps  = len(final_schedule_lohi)
    frontier = state["frontier"]
    rounds   = state["round"]
    epochs   = state["epochs_done"]

    print(f"Run: {run_dir}")
    print(f"  Rounds completed : {rounds}")
    print(f"  Epochs done      : {epochs}")
    print(f"  Final schedule   : {n_steps} steps")
    for i, s in enumerate(final_schedule_lohi.tolist()):
        print(f"    t={i}: sigma={s:.6f}")
    print(f"  Frontier         : {frontier:.6f}")
    print(f"  Patience counter : {state['patience_counter']}")

    # Find last checkpoint
    ckpts = sorted(run_dir.glob("checkpoint_round_*.pth"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpts:
        print(f"ERROR: no checkpoint_round_*.pth found in {run_dir}")
        sys.exit(1)
    last_ckpt = ckpts[-1]
    print(f"\nLoading checkpoint: {last_ckpt.name}")

    model = build_and_load_model(last_ckpt, final_schedule_lohi, args.device)
    print(f"Model loaded ({n_steps}-step final schedule on {args.device})")

    # ---- Trajectory rollout ----
    print(f"\nRunning trajectory rollout ({args.rollout_steps} steps) ...")
    traj_loader = make_traj_loader(args.batch_size, args.rollout_steps)
    traj_metrics = compute_trajectory_metrics(model, traj_loader, args.device, args.rollout_steps)

    print(f"  step1_mse       = {traj_metrics['step1_mse']:.4e}")
    print(f"  avg_mse         = {traj_metrics['avg_mse']:.4e}")
    print(f"  last_step_mse   = {traj_metrics['last_step_mse']:.4e}")
    print(f"  time_to_failure = {traj_metrics['time_to_failure']:.1f}")

    # ---- Per-level errors ----
    print(f"\nComputing per-level errors ({args.n_eval_batches} batches) ...")
    val_loader = make_val_loader(args.batch_size)
    sigmas, clean_mses, inf_mses = compute_per_level_errors(
        model, val_loader, args.device, final_schedule_lohi, n_batches=args.n_eval_batches,
    )

    print(f"  Per-level errors (low→high sigma):")
    for sigma, c, i in zip(sigmas, clean_mses, inf_mses):
        ratio = i / c if (np.isfinite(c) and c > 0) else float("nan")
        print(f"    sigma={sigma:.6f}  clean={c:.3e}  inf={i:.3e}  ratio={ratio:.4f}")

    frontier_clean = clean_mses[0]
    frontier_inf   = inf_mses[0]
    reb = frontier_inf / frontier_clean if (np.isfinite(frontier_clean) and frontier_clean > 0) else None
    print(f"\n  Frontier REB = {reb:.4f}" if reb is not None else "\n  Frontier REB = N/A")

    # ---- Save results ----
    results = {
        "run_dir":        str(run_dir),
        "checkpoint":     last_ckpt.name,
        "rounds":         rounds,
        "epochs_done":    epochs,
        "n_steps":        n_steps,
        "final_schedule": final_schedule_lohi.tolist(),
        "frontier":       frontier,
        "nfe":            n_steps,
        **traj_metrics,
        "reb": reb,
        "per_level": [
            {"sigma": s, "clean_mse": c, "inf_mse": i,
             "ratio": i / c if (np.isfinite(c) and c > 0) else None}
            for s, c, i in zip(sigmas, clean_mses, inf_mses)
        ],
    }

    out_path = run_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
