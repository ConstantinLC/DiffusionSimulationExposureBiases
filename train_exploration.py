"""
Phase 1: Exploration

Trains on a dense log-uniform schedule, starting with the K largest noise
levels and progressively unlocking lower ones as upper levels are solved
(B_own(sigma) <= tau). Saves a per-level checkpoint for each solved sigma
so Phase 2 (greedy schedule construction) can reuse them.

Usage:
    python train_exploration.py +experiment=kolmo_exploration

Expected config keys (under `training`):
    tau                  : float  (default 1.05)
    log_sigma_min        : float  (default -3.0)
    log_sigma_max        : float  (default -0.0001)
    n_exploration_levels : int    (default 40)
    n_active_start       : int    (default 10)   -- K largest sigmas to start with
    epochs_per_pass      : int    (default 50)   -- training epochs between evaluations
    eval_every_epoch     : int    (default 0)    -- evaluate B_own every N epochs within a pass (0=only at end)
    max_passes           : int    (default 30)   -- hard stop on number of passes
    n_eval_batches       : int    (default 30)   -- val batches used to estimate B_own
    learning_rate_start  : float  (default 1e-4)
    learning_rate_end    : float  (default 1e-6)
"""

import os
import json
import logging
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import DataConfig, ExperimentConfig
from src.data.loaders import get_data_loaders
from train import build_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# B_own evaluation
# ---------------------------------------------------------------------------

def compute_b_own(model, val_loader, device, n_batches=30):
    """
    Compute B_own(sigma_t) = mean_i [ E_own_i / E_clean_i ] for every noise
    level currently active in the model's schedule.

    Returns
    -------
    sigmas      : (T,) tensor, noise levels (low → high, matching t=0..T-1)
    mean_ratios : list of float, B_own per level
    mean_cleans : list of float, mean E_clean per level
    """
    model.eval()
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()
    T = len(sigmas)

    all_clean = [[] for _ in range(T)]
    all_ratio = [[] for _ in range(T)]
    spatial_dims = None

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            if batch_idx >= n_batches:
                break

            data   = sample['data'].to(device)
            cond   = data[:, 0]
            target = data[:, 1]

            if spatial_dims is None:
                spatial_dims = tuple(range(1, target.ndim))

            _, ests_clean = model(conditioning=cond, data=target,
                                  return_x0_estimate=True, input_type='clean')
            _, ests_own   = model(conditioning=cond, data=target,
                                  return_x0_estimate=True, input_type='own-pred')

            for t_idx, (est_c, est_o) in enumerate(zip(ests_clean, ests_own)):
                clean_mse = (est_c - target).pow(2).mean(dim=spatial_dims)
                own_mse   = (est_o - target).pow(2).mean(dim=spatial_dims)
                valid     = clean_mse > 0
                ratio     = (own_mse / clean_mse.clamp(min=1e-12))[valid]
                all_clean[t_idx].append(clean_mse[valid].cpu())
                all_ratio[t_idx].append(ratio.cpu())

    mean_ratios, mean_cleans = [], []
    for t_idx in range(T):
        if all_clean[t_idx]:
            mean_cleans.append(torch.cat(all_clean[t_idx]).mean().item())
            mean_ratios.append(torch.cat(all_ratio[t_idx]).mean().item())
        else:
            mean_cleans.append(float('nan'))
            mean_ratios.append(float('nan'))

    return sigmas, mean_ratios, mean_cleans


# ---------------------------------------------------------------------------
# Single training pass
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for sample in train_loader:
        data   = sample['data'].to(device)
        cond   = data[:, 0]
        target = data[:, 1]
        optimizer.zero_grad()
        noise, pred_noise = model(cond, target)
        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # --- Hyper-parameters ---
    tr = cfg.training
    device              = torch.device(tr.get('device', 'cuda'))
    tau                 = float(tr.get('tau', 1.05))
    log_sigma_min       = float(tr.get('log_sigma_min', -3.0))
    log_sigma_max       = float(tr.get('log_sigma_max', -0.0001))
    n_levels            = int(tr.get('n_exploration_levels', 40))
    epochs_per_pass     = int(tr.get('epochs_per_pass', 100))
    eval_every_epoch    = int(tr.get('eval_every_epoch', 0))
    max_passes          = int(tr.get('max_passes', 30))
    n_eval_batches      = int(tr.get('n_eval_batches', 30))

    n_active_start      = int(tr.get('n_active_start', min(10, n_levels)))
    lr_start            = float(tr.get('learning_rate_start', 1e-4))
    lr_end              = float(tr.get('learning_rate_end', 1e-6))
    checkpoint_dir      = cfg.checkpoint_dir
    run_idx = 0
    while os.path.exists(os.path.join(checkpoint_dir, f'run_{run_idx}')):
        run_idx += 1
    checkpoint_dir = os.path.join(checkpoint_dir, f'run_{run_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Cosine exploration schedule (dense near both extremes) ---
    # Map uniform t in [0, 1] through cosine to cluster points near
    # low-noise (sigma_min) and high-noise (sigma_max) ends.
    t = np.linspace(0, 1, n_levels)
    cosine_t = 0.5 * (1 - np.cos(np.pi * t))  # S-curve: dense at 0 and 1
    log_sigmas = log_sigma_min + (log_sigma_max - log_sigma_min) * cosine_t
    all_sigmas = torch.tensor(10.0 ** log_sigmas, dtype=torch.float32)  # (N,) low→high

    log.info(f"Exploration schedule: {n_levels} levels, "
             f"sigma in [{10**log_sigma_min:.2e}, {10**log_sigma_max:.2e}], tau={tau}, "
             f"starting with {n_active_start} largest levels")

    # --- Data ---
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    train_loader, val_loader, _ = get_data_loaders(data_cfg)

    # --- Model: build from config, then override with exploration schedule ---
    experiment_cfg = ExperimentConfig(**OmegaConf.to_container(cfg, resolve=True))
    model = build_model(experiment_cfg)
    model.compute_schedule_variables(all_sigmas)
    model = model.to(device)

    # --- Optimiser (single instance, warm-restarted per pass) ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs_per_pass,
        eta_min=lr_end,
    )
    criterion = torch.nn.MSELoss()

    # --- Exploration state ---
    # all_sigmas is ordered low→high; we start with the n_active_start largest
    # and unlock lower levels one-for-one as upper levels are solved.
    active_mask         = torch.zeros(n_levels, dtype=torch.bool)
    active_mask[-n_active_start:] = True
    # next_to_activate: index of the next sigma to unlock (moves downward from
    # n_levels - n_active_start - 1 toward 0; -1 means nothing left to unlock)
    next_to_activate    = n_levels - n_active_start - 1

    solved        = {}   # sigma_value (float) → {checkpoint, clean_error, ratio, pass}
    state_path    = os.path.join(checkpoint_dir, 'exploration_state.json')
    patience              = int(tr.get('exploration_patience', 5))  # passes without progress before stopping
    passes_without_progress = 0

    # -----------------------------------------------------------------------
    for pass_idx in range(max_passes):

        n_active       = active_mask.sum().item()
        active_sigmas  = all_sigmas[active_mask]

        if n_active == 0 and next_to_activate < 0:
            log.info("All noise levels solved. Stopping.")
            break

        n_pending = next_to_activate + 1
        log.info(f"\n=== Pass {pass_idx + 1}/{max_passes}  |  active: {n_active}  |  "
                 f"pending: {n_pending}  |  solved: {len(solved)} ===")

        # Update model schedule to active levels only and reset LR scheduler
        model.compute_schedule_variables(active_sigmas.to(device))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_start
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_pass, eta_min=lr_end)

        # -- Train for epochs_per_pass epochs, with optional mid-pass evals --
        solved_this_pass = set()  # sigma values solved during this pass

        for epoch in range(epochs_per_pass):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                log.info(f"  epoch {epoch + 1}/{epochs_per_pass}  loss={loss:.6f}")

            # Mid-pass evaluation: register checkpoints and immediately remove solved levels
            is_last_epoch = (epoch + 1) == epochs_per_pass
            do_mid_eval = (eval_every_epoch > 0
                           and (epoch + 1) % eval_every_epoch == 0
                           and not is_last_epoch)
            if do_mid_eval:
                log.info(f"  -- mid-pass eval at epoch {epoch + 1} --")
                sigmas_eval, mean_ratios, mean_cleans = compute_b_own(
                    model, val_loader, device, n_batches=n_eval_batches
                )
                newly_solved_mid = 0
                for sigma, ratio, clean in zip(sigmas_eval, mean_ratios, mean_cleans):
                    sigma_val = sigma.item()
                    status = "SOLVED" if ratio <= tau else f"ratio={ratio:.3f}"
                    log.info(f"  sigma={sigma_val:.5f}  B_own={ratio:.4f}  "
                             f"E_clean={clean:.3e}  [{status}]")
                    if np.isnan(ratio):
                        continue
                    if ratio <= tau and sigma_val not in solved:
                        ckpt_name = f"checkpoint_sigma_{sigma_val:.6f}.pth"
                        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                        torch.save(model.state_dict(), ckpt_path)
                        solved[sigma_val] = {
                            'checkpoint' : ckpt_path,
                            'clean_error': clean,
                            'ratio'      : ratio,
                            'pass'       : pass_idx + 1,
                        }
                        solved_this_pass.add(sigma_val)
                        global_idx = (all_sigmas - torch.tensor(sigma_val)).abs().argmin().item()
                        active_mask[global_idx] = False
                        newly_solved_mid += 1
                if newly_solved_mid > 0:
                    n_unlocked_mid = 0
                    for _ in range(newly_solved_mid):
                        if next_to_activate >= 0:
                            active_mask[next_to_activate] = True
                            next_to_activate -= 1
                            n_unlocked_mid += 1
                    active_sigmas = all_sigmas[active_mask]
                    model.compute_schedule_variables(active_sigmas.to(device))
                    log.info(f"  Removed {newly_solved_mid} solved level(s) mid-pass, "
                             f"unlocked {n_unlocked_mid} new; "
                             f"{active_mask.sum().item()} active.")

        # -- End-of-pass evaluation: register checkpoints AND remove solved levels --
        sigmas_eval, mean_ratios, mean_cleans = compute_b_own(
            model, val_loader, device, n_batches=n_eval_batches
        )

        for sigma, ratio, clean in zip(sigmas_eval, mean_ratios, mean_cleans):
            sigma_val = sigma.item()
            if np.isnan(ratio):
                continue
            status = "SOLVED" if ratio <= tau else f"ratio={ratio:.3f}"
            log.info(f"  sigma={sigma_val:.5f}  B_own={ratio:.4f}  "
                     f"E_clean={clean:.3e}  [{status}]")

            if ratio <= tau and sigma_val not in solved:
                ckpt_name = f"checkpoint_sigma_{sigma_val:.6f}.pth"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                torch.save(model.state_dict(), ckpt_path)
                solved[sigma_val] = {
                    'checkpoint' : ckpt_path,
                    'clean_error': clean,
                    'ratio'      : ratio,
                    'pass'       : pass_idx + 1,
                }
                solved_this_pass.add(sigma_val)

        # Remove all levels solved during this pass from the active schedule
        newly_solved = 0
        for sigma_val in solved_this_pass:
            sigma_t = torch.tensor(sigma_val)
            global_idx = (all_sigmas - sigma_t).abs().argmin().item()
            active_mask[global_idx] = False
            newly_solved += 1

        # Unlock one lower level for each level solved this pass
        n_unlocked = 0
        for _ in range(newly_solved):
            if next_to_activate >= 0:
                active_mask[next_to_activate] = True
                next_to_activate -= 1
                n_unlocked += 1

        # Persist state after every pass
        with open(state_path, 'w') as f:
            json.dump(
                {str(k): v for k, v in sorted(solved.items())},
                f, indent=2
            )

        log.info(f"Pass {pass_idx + 1} done: {newly_solved} newly solved, "
                 f"{n_unlocked} new level(s) unlocked, "
                 f"{active_mask.sum().item()} active, "
                 f"{next_to_activate + 1} pending.")

        if newly_solved == 0:
            passes_without_progress += 1
            log.info(f"No progress this pass ({passes_without_progress}/{patience}).")
            if passes_without_progress >= patience:
                log.info("Patience exhausted. Exploration complete.")
                break
        else:
            passes_without_progress = 0

    # -----------------------------------------------------------------------
    log.info(f"\nExploration finished. {len(solved)} / {n_levels} levels solved.")
    log.info(f"State saved to {state_path}")


if __name__ == '__main__':
    main()
