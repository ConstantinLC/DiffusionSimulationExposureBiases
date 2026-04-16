"""
Phase 1: Exploration (bias-propagation variant)

Builds a *final* schedule greedily from high noise to low noise. A level sigma
is a candidate when the two-step chaining bias from the current schedule
frontier satisfies:

    R1 = B_2step(sigma) / B_own(sigma) < tau

current_pretender tracks the lowest such sigma found so far. When a training
pass yields no level with R1 < tau, the pretender is committed to the final
schedule, the bias is re-evaluated against the new frontier, and current levels
are replenished with unexplored levels below the new frontier.

Usage:
    python train_exploration.py +experiment=kolmo_exploration

Expected config keys (under `training`):
    tau                  : float  (default 1.05)
    log_sigma_min        : float  (default -3.0)
    log_sigma_max        : float  (default -0.0001)
    n_exploration_levels : int    (default 40)
    n_active_start       : int    (default 10)   -- active window size
    epochs_per_pass      : int    (default 50)   -- training epochs between evaluations
    eval_every_epoch     : int    (default 0)    -- log bias every N epochs within pass (0=end only)
    max_passes           : int    (default 30)   -- hard stop on number of passes
    n_eval_batches       : int    (default 30)   -- val batches used to estimate bias
    learning_rate_start  : float  (default 1e-4)
    learning_rate_end    : float  (default 1e-6)
    exploration_patience : int    (default 5)    -- passes without a commit before stopping
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
# B_2step evaluation
# ---------------------------------------------------------------------------

def compute_b_2step(model, val_loader, device, first_step_t_idx, n_batches=30):
    """
    Compute B_2step(sigma_t) for every noise level currently active.

    Two-step sequence: first_step (one denoising step with clean input) -> sigma_t.

      1. At first_step_t_idx, forward-noise the clean target and predict x0_hat_1.
      2. Re-noise x0_hat_1 to sigma_t and predict x0_hat_2.
      3. B_2step(t) = E[||x0_hat_2 - target||^2] / E[||x0_clean(t) - target||^2]

    Parameters
    ----------
    first_step_t_idx : int
        Index in the model's current schedule to use as the first denoising
        step. Should correspond to the smallest level in the final schedule.

    Returns
    -------
    sigmas      : (T,) tensor, noise levels (low → high)
    mean_ratios : list of float, B_2step per level
    mean_cleans : list of float, mean E_clean per level
    """
    model.eval()
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()
    T      = len(sigmas)

    sqrtA   = model.sqrtAlphasCumprod
    sqrtOMA = model.sqrtOneMinusAlphasCumprod
    C_cond  = model.condChannels

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
            B      = target.shape[0]

            if spatial_dims is None:
                spatial_dims = tuple(range(1, target.ndim))

            # --- Step 1: one denoising step at first_step_t_idx with clean input ---
            t_first  = torch.full((B,), first_step_t_idx, device=device, dtype=torch.long)
            d_first  = sqrtA[t_first] * target + sqrtOMA[t_first] * torch.randn_like(target)
            inp      = torch.cat((cond, d_first), dim=1)
            eps_hat  = model.unet(inp, t_first)[:, C_cond:]
            x0_hat_1 = (d_first - sqrtOMA[t_first] * eps_hat) / sqrtA[t_first].clamp(min=1e-8)

            # --- Step 2: for each t, re-noise x0_hat_1 and predict ---
            for t_idx in range(T):
                t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)

                # Clean baseline at level t
                d_clean  = sqrtA[t_vec] * target + sqrtOMA[t_vec] * torch.randn_like(target)
                inp_c    = torch.cat((cond, d_clean), dim=1)
                eps_c    = model.unet(inp_c, t_vec)[:, C_cond:]
                x0_clean = (d_clean - sqrtOMA[t_vec] * eps_c) / sqrtA[t_vec].clamp(min=1e-8)

                # Two-step: re-noise x0_hat_1 to level t, then predict
                d_2step   = sqrtA[t_vec] * x0_hat_1 + sqrtOMA[t_vec] * torch.randn_like(target)
                inp_2     = torch.cat((cond, d_2step), dim=1)
                eps_2     = model.unet(inp_2, t_vec)[:, C_cond:]
                x0_2step  = (d_2step - sqrtOMA[t_vec] * eps_2) / sqrtA[t_vec].clamp(min=1e-8)

                clean_mse   = (x0_clean - target).pow(2).mean(dim=spatial_dims)
                twostep_mse = (x0_2step - target).pow(2).mean(dim=spatial_dims)
                valid       = clean_mse > 0
                ratio       = (twostep_mse / clean_mse.clamp(min=1e-12))[valid]
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
# check_bias: evaluate R1 and update pretender + active_mask
# ---------------------------------------------------------------------------

def apply_check_bias(sigmas_eval, b_own, b_2step, tau,
                     all_sigmas, active_mask, final_schedule,
                     current_pretender, min_sigma=None, label=""):
    """
    Iterate sigmas in decreasing order, apply the R1 < tau criterion, update
    current_pretender and active_mask in place.

    Parameters
    ----------
    min_sigma : float or None
        If set, only check levels strictly smaller than this value.
    label     : str
        Prefix for log lines (e.g. "[post-commit]").

    Returns
    -------
    current_pretender : float or None (updated)
    any_passed        : bool
    """
    any_passed = False
    final_set  = set(final_schedule)

    for t_idx in reversed(range(len(sigmas_eval))):
        sigma_val = sigmas_eval[t_idx].item()
        if min_sigma is not None and sigma_val >= min_sigma:
            continue

        ob = b_own[t_idx]
        ts = b_2step[t_idx]
        if np.isnan(ob) or np.isnan(ts) or ob <= 1e-12:
            continue

        ratio = ts / ob
        status = "PASS" if ratio < tau else "fail"
        log.info(f"  {label}sigma={sigma_val:.5f}  B_own={ob:.4f}  "
                 f"B_2step={ts:.4f}  R1={ratio:.4f}  [{status}]")

        if ratio < tau and sigma_val not in final_set and (current_pretender is None or sigma_val < current_pretender):
            current_pretender = sigma_val
            any_passed = True
            log.info(f"  {label}>>> pretender updated to sigma={sigma_val:.5f}")
            # Remove larger active levels that are not in the final schedule
            for i in range(len(all_sigmas)):
                if (active_mask[i]
                        and all_sigmas[i].item() > sigma_val
                        and all_sigmas[i].item() not in final_set):
                    active_mask[i] = False

    return current_pretender, any_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # --- Hyper-parameters ---
    tr = cfg.training
    torch.set_num_threads(int(tr.get('num_cpu_threads', 4)))
    device           = torch.device(tr.get('device', 'cuda'))
    tau              = float(tr.get('tau', 1.05))
    log_sigma_min    = float(tr.get('log_sigma_min', -3.0))
    log_sigma_max    = float(tr.get('log_sigma_max', -0.0001))
    n_levels         = int(tr.get('n_exploration_levels', 40))
    epochs_per_pass  = int(tr.get('epochs_per_pass', 100))
    eval_every_epoch = int(tr.get('eval_every_epoch', 0))
    max_passes       = int(tr.get('max_passes', 30))
    n_eval_batches   = int(tr.get('n_eval_batches', 30))
    n_active_start   = int(tr.get('n_active_start', min(10, n_levels)))
    lr_start         = float(tr.get('learning_rate_start', 1e-4))
    lr_end           = float(tr.get('learning_rate_end', 1e-6))
    patience         = int(tr.get('exploration_patience', 5))
    checkpoint_dir   = cfg.checkpoint_dir

    run_idx = 0
    while os.path.exists(os.path.join(checkpoint_dir, f'run_{run_idx}')):
        run_idx += 1
    checkpoint_dir = os.path.join(checkpoint_dir, f'run_{run_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Cosine exploration schedule (dense near both extremes) ---
    t        = np.linspace(0, 1, n_levels)
    cosine_t = 0.5 * (1 - np.cos(np.pi * t))
    log_sigmas = log_sigma_min + (log_sigma_max - log_sigma_min) * cosine_t
    all_sigmas = torch.tensor(10.0 ** log_sigmas, dtype=torch.float32)  # (N,) low→high

    log.info(f"Exploration schedule: {n_levels} levels, "
             f"sigma in [{10**log_sigma_min:.2e}, {10**log_sigma_max:.2e}], "
             f"tau={tau}, window size={n_active_start}")

    # --- Data ---
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    train_loader, val_loader, _ = get_data_loaders(data_cfg)

    # --- Model ---
    experiment_cfg = ExperimentConfig(**OmegaConf.to_container(cfg, resolve=True))
    model = build_model(experiment_cfg)
    model.compute_schedule_variables(all_sigmas)
    model = model.to(device)

    # --- Optimiser ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_pass, eta_min=lr_end)
    criterion = torch.nn.MSELoss()

    # --- Algorithm state ---
    # active_mask: which indices in all_sigmas are in "current levels"
    active_mask = torch.zeros(n_levels, dtype=torch.bool)
    active_mask[-n_active_start:] = True
    # next_to_explore: index of the next level to add (moves downward)
    next_to_explore = n_levels - n_active_start - 1

    # final_schedule: sigma values committed, stored high→low
    final_schedule    = [all_sigmas[-1].item()]
    current_pretender = None
    state_path        = os.path.join(checkpoint_dir, 'exploration_state.json')
    passes_without_commit = 0

    log.info(f"Initial final schedule: [{all_sigmas[-1].item():.5f}]")

    # -----------------------------------------------------------------------
    for pass_idx in range(max_passes):

        active_sigmas = all_sigmas[active_mask]
        n_active      = active_mask.sum().item()

        if n_active == 0:
            log.info("No active levels. Stopping.")
            break

        pretender_str = f"{current_pretender:.5f}" if current_pretender is not None else "None"
        log.info(f"\n=== Pass {pass_idx + 1}/{max_passes}  |  active: {n_active}  |  "
                 f"final: {len(final_schedule)}  |  pretender: {pretender_str} ===")

        # Update model schedule and reset LR
        model.compute_schedule_variables(active_sigmas.to(device))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_start
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_pass, eta_min=lr_end)

        # --- Train, evaluating and updating state every eval_every_epoch epochs ---
        stop_early = False
        for epoch in range(epochs_per_pass):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                log.info(f"  epoch {epoch + 1}/{epochs_per_pass}  loss={loss:.6f}")

            is_last_epoch = (epoch + 1) == epochs_per_pass
            do_eval = (eval_every_epoch > 0 and (epoch + 1) % eval_every_epoch == 0) \
                      or (eval_every_epoch == 0 and is_last_epoch)
            if not do_eval:
                continue

            log.info(f"  -- eval at epoch {epoch + 1} --")
            sigmas_eval, b_own, _ = compute_b_own(
                model, val_loader, device, n_batches=n_eval_batches)

            min_final      = min(final_schedule)
            first_step_idx = (sigmas_eval - min_final).abs().argmin().item()
            _, b_2step, _  = compute_b_2step(
                model, val_loader, device, first_step_idx, n_batches=n_eval_batches)

            current_pretender, any_passed = apply_check_bias(
                sigmas_eval, b_own, b_2step, tau,
                all_sigmas, active_mask, final_schedule, current_pretender)

            # Commit pretender when no level passes
            if not any_passed and current_pretender is not None:
                committed = current_pretender
                log.info(f"  No improvement → committing {committed:.5f} to final schedule")

                final_schedule.append(committed)
                final_schedule.sort(reverse=True)

                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_sigma_{committed:.6f}.pth")
                torch.save(model.state_dict(), ckpt_path)
                log.info(f"  Checkpoint saved: {ckpt_path}")

                # Re-evaluate with new frontier (first step = committed level)
                current_pretender = None
                new_fst_idx = (sigmas_eval - committed).abs().argmin().item()
                _, b_2step_new, _ = compute_b_2step(
                    model, val_loader, device, new_fst_idx, n_batches=n_eval_batches)

                current_pretender, _ = apply_check_bias(
                    sigmas_eval, b_own, b_2step_new, tau,
                    all_sigmas, active_mask, final_schedule, current_pretender,
                    min_sigma=committed, label="[post-commit] ")

                passes_without_commit = 0

            elif not any_passed:
                passes_without_commit += 1
                log.info(f"  No pretender and no pass ({passes_without_commit}/{patience}).")
                if passes_without_commit >= patience:
                    stop_early = True
                    break

            else:
                passes_without_commit = 0

            # Replenish current levels to window size
            n_added = 0
            while active_mask.sum().item() < n_active_start and next_to_explore >= 0:
                active_mask[next_to_explore] = True
                next_to_explore -= 1
                n_added += 1
            if n_added:
                log.info(f"  Replenished {n_added} level(s); "
                         f"{active_mask.sum().item()} active, "
                         f"{next_to_explore + 1} unexplored.")

            # Update model schedule so remaining epochs train on the trimmed+replenished levels
            model.compute_schedule_variables(all_sigmas[active_mask].to(device))

            # Persist state
            with open(state_path, 'w') as f:
                json.dump({
                    'final_schedule'  : final_schedule,
                    'pretender'       : current_pretender,
                    'n_active'        : active_mask.sum().item(),
                    'next_to_explore' : next_to_explore,
                }, f, indent=2)

            log.info(f"  Final schedule: {[f'{s:.5f}' for s in sorted(final_schedule)]}")

        log.info(f"Pass {pass_idx + 1} done.")
        if stop_early:
            log.info("Patience exhausted. Exploration complete.")
            break

    # -----------------------------------------------------------------------
    log.info(f"\nExploration finished. Final schedule ({len(final_schedule)} levels): "
             f"{[f'{s:.5f}' for s in sorted(final_schedule)]}")
    log.info(f"State saved to {state_path}")


if __name__ == '__main__':
    main()
