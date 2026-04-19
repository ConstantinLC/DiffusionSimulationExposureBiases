"""
Phase 2: Greedy Schedule Construction

Given a completed Phase 1 exploration run (per-level checkpoints), constructs
the diffusion schedule greedily by starting at sigma_0 = min(solved) and
iteratively picking the largest feasible sigma' such that B^(2S) <= tau.

Usage:
    python train_greedy_schedule.py +experiment=ks_exploration \
        training.exploration_run_dir=./checkpoints/KuramotoSivashinsky/exploration/run_1

Expected config keys (under `training`):
    exploration_run_dir  : str   (path to Phase 1 run folder)
    tau                  : float (default 1.05)
    n_eval_batches       : int   (default 30)
"""

import os
import json
import logging
import numpy as np

import torch
import wandb
from torch import nn

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import DataConfig, ExperimentConfig
from src.data.loaders import get_data_loaders
from src.training.diffusion_trainer import train_diffusion_model
from src.utils.general import get_run_dir_name
from train import build_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Two-step bias evaluation
# ---------------------------------------------------------------------------

def compute_two_step_bias_cross_checkpoint(
    model, checkpoint_high, checkpoint_low,
    sigma_high, sigma_low,
    val_loader, device, n_batches=30, n_noise_samples=1,
):
    """
    Compute B^(2S)_{theta*(sigma_high), theta*(sigma_low)}(sigma_low, sigma_high).

    Uses checkpoint_high for denoising at sigma_high (first step) and
    checkpoint_low for denoising at sigma_low (second step).

    The model schedule is set to [sigma_low, sigma_high] (indices 0 and 1).
    Checkpoint swaps happen once per phase (not per batch) for efficiency.
    n_noise_samples independent noise draws are averaged per (sample, phase).
    """
    model.eval()
    schedule = torch.tensor([sigma_low, sigma_high], dtype=torch.float32)
    IDX_LOW, IDX_HIGH = 0, 1

    # Schedule buffers change shape per evaluation pair; strip them so load_state_dict
    # doesn't raise on size mismatch — compute_schedule_variables sets them below.
    _SCHEDULE_KEYS = {'sqrtAlphasCumprod', 'sqrtOneMinusAlphasCumprod', 'unet.sigmas'}

    def _load_weights(ckpt):
        model.load_state_dict({k: v for k, v in ckpt.items() if k not in _SCHEDULE_KEYS}, strict=False)
        model.compute_schedule_variables(schedule)

    # --- Phase A: load checkpoint_high, denoise all batches at sigma_high ---
    _load_weights(checkpoint_high)
    model = model.to(device)

    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze()
    sqrt_alpha = model.sqrtAlphasCumprod.squeeze()

    # For each batch, store n_noise_samples x0_hat draws
    x0_hats = []   # list of (n_noise_samples, N, ...) tensors
    targets_list = []
    conds_list = []
    spatial_dims = None

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            if batch_idx >= n_batches:
                break

            data = sample['data'].to(device)
            cond = data[:, 0]
            target = data[:, 1]

            if spatial_dims is None:
                spatial_dims = tuple(range(1, target.ndim))

            N = target.shape[0]
            t_high = torch.full((N,), IDX_HIGH, device=device, dtype=torch.long)

            x0_hat_acc = torch.zeros_like(target)
            for _ in range(n_noise_samples):
                eps_high = torch.randn_like(target)
                y_noisy_high = sqrt_alpha[t_high] * target + sigmas[t_high] * eps_high
                inp_high = torch.cat((cond, y_noisy_high), dim=1)
                pred_noise_high = model.unet(inp_high, t_high)[:, cond.shape[1]:]
                x0_hat_acc += (y_noisy_high - sigmas[t_high] * pred_noise_high) / sqrt_alpha[t_high]
            x0_hat_acc /= n_noise_samples

            x0_hats.append(x0_hat_acc.cpu())
            targets_list.append(target.cpu())
            conds_list.append(cond.cpu())

    # --- Phase B: load checkpoint_low, denoise at sigma_low ---
    _load_weights(checkpoint_low)
    model = model.to(device)

    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze()
    sqrt_alpha = model.sqrtAlphasCumprod.squeeze()

    all_clean_mse = []
    all_twostep_mse = []

    with torch.no_grad():
        for x0_hat, target, cond in zip(x0_hats, targets_list, conds_list):
            x0_hat = x0_hat.to(device)
            target = target.to(device)
            cond = cond.to(device)

            N = target.shape[0]
            t_low = torch.full((N,), IDX_LOW, device=device, dtype=torch.long)

            twostep_acc = torch.zeros(N, device=device)
            clean_acc = torch.zeros(N, device=device)
            for _ in range(n_noise_samples):
                # Re-noise x0_hat to sigma_low
                eps_low = torch.randn_like(target)
                y_noisy_low = sqrt_alpha[t_low] * x0_hat + sigmas[t_low] * eps_low
                inp_low = torch.cat((cond, y_noisy_low), dim=1)
                pred_noise_low = model.unet(inp_low, t_low)[:, cond.shape[1]:]
                x0_final = (y_noisy_low - sigmas[t_low] * pred_noise_low) / sqrt_alpha[t_low]
                twostep_acc += (x0_final - target).pow(2).mean(dim=spatial_dims)

                # Clean error at sigma_low
                eps_clean = torch.randn_like(target)
                y_clean_low = sqrt_alpha[t_low] * target + sigmas[t_low] * eps_clean
                inp_clean = torch.cat((cond, y_clean_low), dim=1)
                pred_noise_clean = model.unet(inp_clean, t_low)[:, cond.shape[1]:]
                x0_clean = (y_clean_low - sigmas[t_low] * pred_noise_clean) / sqrt_alpha[t_low]
                clean_acc += (x0_clean - target).pow(2).mean(dim=spatial_dims)

            all_twostep_mse.append((twostep_acc / n_noise_samples).cpu())
            all_clean_mse.append((clean_acc / n_noise_samples).cpu())

    mean_twostep = torch.cat(all_twostep_mse).mean().item()
    mean_clean = torch.cat(all_clean_mse).mean().item()
    bias = mean_twostep / max(mean_clean, 1e-12)
    return bias, mean_clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    tr = cfg.training
    device = torch.device(tr.get('device', 'cuda'))
    seed = tr.get('seed', None)
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    tau = float(tr.get('tau', 1.05))
    n_eval_batches = int(tr.get('n_eval_batches', 30))
    n_noise_samples = int(tr.get('n_noise_samples', 1))
    run_dir = str(tr.exploration_run_dir)

    # --- Load exploration state ---
    state_path = os.path.join(run_dir, 'exploration_state.json')
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"No exploration_state.json found in {run_dir}")

    with open(state_path) as f:
        solved_state = json.load(f)

    # Parse solved sigmas (sorted low -> high)
    solved_sigmas = sorted([float(s) for s in solved_state.keys()])
    log.info(f"Loaded {len(solved_sigmas)} solved levels from {state_path}")
    log.info(f"Sigma range: [{solved_sigmas[0]:.6f}, {solved_sigmas[-1]:.6f}]")

    # --- Data ---
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    train_loader, val_loader, traj_loader = get_data_loaders(data_cfg)

    # --- Build model skeleton (weights will be loaded from checkpoints) ---
    experiment_cfg = ExperimentConfig(**OmegaConf.to_container(cfg, resolve=True))
    model = build_model(experiment_cfg)
    model = model.to(device)

    # --- Load all checkpoints into memory ---
    log.info("Loading checkpoints...")
    checkpoints = {}
    for sigma_val in solved_sigmas:
        info = solved_state[str(sigma_val)]
        ckpt_path = info['checkpoint']
        # Handle relative paths (relative to project root, not run_dir)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.getcwd(), ckpt_path)
        if not os.path.exists(ckpt_path):
            # Try relative to run_dir
            alt_path = os.path.join(run_dir, os.path.basename(ckpt_path))
            if os.path.exists(alt_path):
                ckpt_path = alt_path
            else:
                log.warning(f"Checkpoint not found for sigma={sigma_val:.6f}: {ckpt_path}")
                continue
        checkpoints[sigma_val] = torch.load(ckpt_path, map_location=device)
    log.info(f"Loaded {len(checkpoints)} checkpoints")

    available_sigmas = sorted([s for s in solved_sigmas if s in checkpoints])
    if len(available_sigmas) < 2:
        raise RuntimeError("Need at least 2 solved levels to construct a schedule")

    sigma_T = available_sigmas[-1]  # maximum noise level

    # -------------------------------------------------------------------
    # Greedy schedule construction (Algorithm 2 from the paper)
    # -------------------------------------------------------------------
    # S = [sigma_0], sigma_0 = min(solved)
    # While sigma_t < sigma_T:
    #   sigma_{t+1} = max { sigma' in solved : B^(2S)_{theta*(sigma'), theta*(sigma_t)}(sigma_t, sigma') <= tau }
    #   S = S ∪ [sigma_{t+1}]
    # -------------------------------------------------------------------

    schedule = [available_sigmas[0]]
    log.info(f"\n{'='*60}")
    log.info(f"Greedy schedule construction  |  tau={tau}  n_noise_samples={n_noise_samples}")
    log.info(f"{'='*60}")
    log.info(f"sigma_0 = {schedule[0]:.6f}  (minimum solved level)")

    t = 0
    while schedule[t] < sigma_T:
        sigma_t = schedule[t]
        ckpt_low = checkpoints[sigma_t]

        # Candidates: all solved sigmas strictly greater than sigma_t
        candidates = [s for s in available_sigmas if s > sigma_t]
        if not candidates:
            log.warning(f"No candidates above sigma_t={sigma_t:.6f}. Stopping.")
            break

        # Search from largest to smallest for greedy max jump
        best_sigma = None
        for sigma_candidate in reversed(candidates):
            ckpt_high = checkpoints[sigma_candidate]

            bias, clean_err = compute_two_step_bias_cross_checkpoint(
                model, ckpt_high, ckpt_low,
                sigma_candidate, sigma_t,
                val_loader, device, n_batches=n_eval_batches,
                n_noise_samples=n_noise_samples,
            )
            log.info(f"  Eval: sigma_t={sigma_t:.6f} -> sigma'={sigma_candidate:.6f}  "
                     f"B^(2S)={bias:.4f}  E_clean={clean_err:.3e}  "
                     f"{'FEASIBLE' if bias <= tau else 'REJECTED'}")

            if bias <= tau:
                best_sigma = sigma_candidate
                break  # greedy: take the largest feasible jump

        if best_sigma is None:
            # No feasible jump found; fall back to the next sigma above
            best_sigma = candidates[0]
            log.warning(f"  No feasible jump from sigma_t={sigma_t:.6f}. "
                        f"Falling back to next level: {best_sigma:.6f}")

        schedule.append(best_sigma)
        t += 1
        log.info(f"  => sigma_{t} = {best_sigma:.6f}  "
                 f"(schedule length: {len(schedule)})")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    output_dir = os.path.join(run_dir, 'greedy_schedule')
    os.makedirs(output_dir, exist_ok=True)

    result = {
        'tau': tau,
        'n_eval_batches': n_eval_batches,
        'schedule': schedule,
        'schedule_log10': [float(np.log10(s)) for s in schedule],
        'n_steps': len(schedule),
        'sigma_min': schedule[0],
        'sigma_max': schedule[-1],
        'source_run': run_dir,
    }

    result_path = os.path.join(output_dir, 'schedule.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"Greedy schedule construction complete")
    log.info(f"{'='*60}")
    log.info(f"Schedule ({len(schedule)} steps):")
    for i, s in enumerate(schedule):
        log.info(f"  t={i}: sigma={s:.6f} (log10={np.log10(s):.3f})")
    log.info(f"Saved to {result_path}")

    # -------------------------------------------------------------------
    # Phase 3: train a fresh diffusion model on the greedy schedule
    # -------------------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info("Phase 3: training diffusion model on greedy schedule")
    log.info(f"{'='*60}")

    # Merge pretraining params into training, and override model schedule
    train_cfg = OmegaConf.merge(cfg, {
        "training": OmegaConf.to_container(cfg.pretraining, resolve=True),
        "model": {
            "diffSchedule": "from_file",
            "schedule_path": result_path,
        },
        "checkpoint_dir": os.path.join(run_dir, "greedy_trained"),
    })

    greedy_config = ExperimentConfig.from_hydra(train_cfg)
    legacy = greedy_config.to_legacy_dict()

    greedy_checkpoint_dir = greedy_config.checkpoint_dir
    os.makedirs(greedy_checkpoint_dir, exist_ok=True)

    # Save config
    with open(os.path.join(greedy_checkpoint_dir, "config.json"), "w") as f:
        json.dump(legacy, f, indent=4)

    # W&B
    run_name = get_run_dir_name(greedy_checkpoint_dir, greedy_config.model)
    wandb.init(
        project=greedy_config.wandb.project,
        entity=greedy_config.wandb.entity,
        name=run_name,
        config=legacy,
        mode="disabled" if cfg.get("debugging", False) else "online",
    )

    # Build fresh model on the greedy schedule
    greedy_model = build_model(greedy_config)
    greedy_model = greedy_model.to(device)

    criterion = nn.MSELoss()

    train_diffusion_model(
        greedy_model,
        train_loader,
        val_loader,
        traj_loader,
        legacy["train_params"],
        criterion,
        legacy,
        greedy_checkpoint_dir,
        device=device,
        is_master=True,
    )

    wandb.finish()


if __name__ == '__main__':
    main()
