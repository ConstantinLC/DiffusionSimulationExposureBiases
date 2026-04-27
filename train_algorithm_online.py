"""
Online Algorithm: Joint Exploration and Schedule Construction

Generates a diffusion schedule simultaneously with model training.  The
exploration schedule is used throughout; the final schedule is built
incrementally as the model learns.

Algorithm:
    1. Build a dense cosine-spaced exploration schedule (sigma_min → sigma_T).
    2. Initialise model on the full exploration schedule.
    3. final_schedule = [sigma_T]; frontier = sigma_T.
    4. For each round (up to max_rounds):
       a. Update model schedule to current active exploration levels.
       b. Train epochs_per_round epochs.
       c. Compute B^2S(sigma, frontier) for every active level below frontier.
       d. Find the *smallest* sigma where B^2S <= tau → sigma_next.
       e. If found:
            - Add sigma_next to final_schedule.
            - Remove exploration levels strictly between sigma_next and frontier
              (these can be skipped in the final schedule).
            - Set frontier = sigma_next.
            - Reset patience counter.
       f. If not found: increment patience counter.
       g. Persist state; stop if patience >= exploration_patience.
    5. Repeat from 4 until no new level is discovered in a round.

Usage:
    python train_algorithm_online.py +experiment=ks_exploration

Expected config keys (under `training`):
    tau                  : float  (default 1.05)
    log_sigma_min        : float  (default -3.0)
    log_sigma_max        : float  (default -0.0001)
    n_exploration_levels : int    (default 40)
    epochs_per_round         : int   (default 100)   -- K epochs per round (base)
    epoch_increase_per_round : int   (default 0)     -- added epochs each successive round
    total_epochs             : int   (default None)  -- stop when cumulative epochs >= this
    max_rounds               : int   (default 100)   -- fallback hard stop when total_epochs unset
    exploration_patience     : int   (default 5)     -- consecutive fruitless rounds before stopping
    n_eval_batches       : int    (default 30)
    n_noise_samples      : int    (default 4)
    learning_rate_start  : float  (default 1e-4)  -- initial LR, reset each round
    learning_rate_end    : float  (default 1e-6)  -- eta_min for cosine annealing
    n_active_window      : int    (default 10)   -- frontier + K-1 closest levels below it
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
# B^2S evaluation from a fixed frontier level
# ---------------------------------------------------------------------------

def compute_b_2step_from_frontier(
    model, val_loader, device, frontier_sigma,
    n_batches=30, n_noise_samples=4,
):
    """
    Compute B^2S(sigma, frontier_sigma) for every active schedule level below
    frontier_sigma.

    The model's current schedule is used as-is.  The first denoising step is
    performed at frontier_sigma; the second step at each candidate level sigma.

    Returns
    -------
    candidate_sigmas : (N,) tensor of sigma values (low → high, all < frontier_sigma)
    mean_ratios      : list[float], B^2S per candidate (same order)
    mean_cleans      : list[float], mean E_clean per candidate
    """
    model.eval()
    sigmas_all = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()  # low→high

    # Index of the frontier in the current schedule
    frontier_t = int((sigmas_all - frontier_sigma).abs().argmin().item())
    n_candidates = frontier_t  # indices 0 .. frontier_t-1

    if n_candidates == 0:
        return sigmas_all[:0], [], []

    sqrtA   = model.sqrtAlphasCumprod          # (T, ...) on device
    sqrtOMA = model.sqrtOneMinusAlphasCumprod  # (T, ...) on device
    C_cond  = model.condChannels

    all_clean = [[] for _ in range(n_candidates)]
    all_ratio = [[] for _ in range(n_candidates)]
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

            # ---- Step 1: denoise at frontier_sigma (averaged over noise draws) ----
            t_high = torch.full((B,), frontier_t, device=device, dtype=torch.long)
            x0_hat_acc = torch.zeros_like(target)
            for _ in range(n_noise_samples):
                eps      = torch.randn_like(target)
                y_noisy  = sqrtA[t_high] * target + sqrtOMA[t_high] * eps
                inp      = torch.cat((cond, y_noisy), dim=1)
                pred     = model.unet(inp, t_high)[:, C_cond:]
                x0_hat_acc += (y_noisy - sqrtOMA[t_high] * pred) / sqrtA[t_high].clamp(min=1e-8)
            x0_hat = x0_hat_acc / n_noise_samples  # (B, ...)

            # ---- Step 2: for each candidate level, re-noise x0_hat and predict ----
            for t_idx in range(n_candidates):
                t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)

                # Two-step prediction
                eps_low  = torch.randn_like(target)
                y_2step  = sqrtA[t_vec] * x0_hat + sqrtOMA[t_vec] * eps_low
                inp_2    = torch.cat((cond, y_2step), dim=1)
                pred_2   = model.unet(inp_2, t_vec)[:, C_cond:]
                x0_2step = (y_2step - sqrtOMA[t_vec] * pred_2) / sqrtA[t_vec].clamp(min=1e-8)

                # Clean-input baseline at this level
                eps_c    = torch.randn_like(target)
                y_clean  = sqrtA[t_vec] * target + sqrtOMA[t_vec] * eps_c
                inp_c    = torch.cat((cond, y_clean), dim=1)
                pred_c   = model.unet(inp_c, t_vec)[:, C_cond:]
                x0_clean = (y_clean - sqrtOMA[t_vec] * pred_c) / sqrtA[t_vec].clamp(min=1e-8)

                twostep_mse = (x0_2step - target).pow(2).mean(dim=spatial_dims)
                clean_mse   = (x0_clean - target).pow(2).mean(dim=spatial_dims)
                valid = clean_mse > 0
                ratio = (twostep_mse / clean_mse.clamp(min=1e-12))[valid]

                all_clean[t_idx].append(clean_mse[valid].cpu())
                all_ratio[t_idx].append(ratio.cpu())

    mean_ratios, mean_cleans = [], []
    for t_idx in range(n_candidates):
        if all_clean[t_idx]:
            mean_cleans.append(torch.cat(all_clean[t_idx]).mean().item())
            mean_ratios.append(torch.cat(all_ratio[t_idx]).mean().item())
        else:
            mean_cleans.append(float('nan'))
            mean_ratios.append(float('nan'))

    return sigmas_all[:n_candidates], mean_ratios, mean_cleans


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
# Per-level clean and inference errors over the final schedule
# ---------------------------------------------------------------------------

def compute_per_level_errors(model, val_loader, device, final_schedule_sigmas_lohi, n_batches=30):
    """
    For each sigma in final_schedule_sigmas_lohi (low→high) compute:
      clean_mse     : single-step denoising MSE when x is noised directly from target
      inference_mse : MSE of x0_hat when x was propagated by the multi-step chain
                      running top→bottom through the schedule

    Caller must restore the model schedule after this call.

    Returns
    -------
    sigmas         : list[float]  (low→high, same order as input)
    clean_mses     : list[float]
    inference_mses : list[float]
    """
    model.eval()
    model.compute_schedule_variables(final_schedule_sigmas_lohi.to(device))

    n_steps = final_schedule_sigmas_lohi.shape[0]
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

            data   = sample['data'].to(device)
            cond   = data[:, 0]
            target = data[:, 1]
            B      = target.shape[0]

            if spatial_dims is None:
                spatial_dims = tuple(range(1, target.ndim))

            # ---- Clean MSE at each level ----
            for t_idx in range(n_steps):
                t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)
                eps   = torch.randn_like(target)
                y     = sqrtA[t_vec] * target + sqrtOMA[t_vec] * eps
                inp   = torch.cat((cond, y), dim=1)
                pred  = model.unet(inp, t_vec)[:, C_cond:]
                x0    = (y - sqrtOMA[t_vec] * pred) / sqrtA[t_vec].clamp(min=1e-8)
                all_clean_mse[t_idx].append((x0 - target).pow(2).mean(dim=spatial_dims).cpu())

            # ---- Inference MSE: run chain top→bottom, record x0_hat at each step ----
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

    sigmas = final_schedule_sigmas_lohi.tolist()
    clean_mses = [
        torch.cat(all_clean_mse[t]).mean().item() if all_clean_mse[t] else float('nan')
        for t in range(n_steps)
    ]
    inf_mses = [
        torch.cat(all_inf_mse[t]).mean().item() if all_inf_mse[t] else float('nan')
        for t in range(n_steps)
    ]
    return sigmas, clean_mses, inf_mses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # --- Hyper-parameters ---
    tr = cfg.training
    torch.set_num_threads(int(tr.get('num_cpu_threads', 4)))
    device = torch.device(tr.get('device', 'cuda'))

    seed = tr.seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    tau                  = float(tr.get('tau', 1.05))
    log_sigma_min        = float(tr.get('log_sigma_min', -3.0))
    log_sigma_max        = float(tr.get('log_sigma_max', -0.0001))
    n_levels             = int(tr.get('n_exploration_levels', 40))
    epochs_per_round          = int(tr.get('epochs_per_round', 100))
    epoch_increase_per_round  = int(tr.get('epoch_increase_per_round', 0))
    _te                       = tr.get('total_epochs', None)
    total_epochs              = int(_te) if _te is not None else None
    max_rounds                = int(tr.get('max_rounds', 100))
    patience_limit       = int(tr.get('exploration_patience', 5))
    n_eval_batches       = int(tr.get('n_eval_batches', 30))
    n_noise_samples      = int(tr.get('n_noise_samples', 4))
    eval_every_epoch     = int(tr.get('eval_every_epoch', 0))
    lr_start             = float(tr.get('learning_rate_start', 1e-4))
    lr_end               = float(tr.get('learning_rate_end', 1e-6))

    n_active_window      = int(tr.get('n_active_window', 10))
    checkpoint_dir       = cfg.checkpoint_dir

    # Auto-create a run_N directory (same logic as train_exploration.py)
    state_exists = os.path.exists(os.path.join(checkpoint_dir, 'online_state.json'))
    if os.path.basename(checkpoint_dir).startswith('run_') and not state_exists:
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        run_idx = 0
        while os.path.exists(os.path.join(checkpoint_dir, f'run_{run_idx}')):
            run_idx += 1
        checkpoint_dir = os.path.join(checkpoint_dir, f'run_{run_idx}')
        os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Cosine exploration schedule (dense near both extremes) ---
    t_uniform = np.linspace(0, 1, n_levels)
    cosine_t  = 0.5 * (1 - np.cos(np.pi * t_uniform))
    log_sigmas = log_sigma_min + (log_sigma_max - log_sigma_min) * cosine_t
    all_sigmas = torch.tensor(10.0 ** log_sigmas, dtype=torch.float32)  # low→high

    sigma_T = all_sigmas[-1].item()

    log.info(
        f"Online algorithm: {n_levels} exploration levels, "
        f"sigma in [{10**log_sigma_min:.2e}, {10**log_sigma_max:.2e}], "
        f"tau={tau}, epochs_per_round={epochs_per_round}"
        + (f" (+{epoch_increase_per_round}/round)" if epoch_increase_per_round else "")
    )

    # --- Data ---
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    train_loader, val_loader, _ = get_data_loaders(data_cfg)

    # --- Model: initialise on the full exploration schedule ---
    experiment_cfg = ExperimentConfig(**OmegaConf.to_container(cfg, resolve=True))
    model = build_model(experiment_cfg)
    model.compute_schedule_variables(all_sigmas)
    model = model.to(device)

    # --- Optimiser ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    criterion = torch.nn.MSELoss()

    # --- State ---
    # exploration_mask[i] = True  →  all_sigmas[i] is still in the exploration schedule
    exploration_mask = torch.ones(n_levels, dtype=torch.bool)

    # final_schedule: list of sigma values, high → low  (sigma_T first)
    final_schedule  = [sigma_T]
    frontier        = sigma_T   # lowest sigma currently in the final schedule

    state_path       = os.path.join(checkpoint_dir, 'online_state.json')
    patience_counter = 0
    epochs_done      = 0

    cosine_T_max = total_epochs if total_epochs is not None else max_rounds * epochs_per_round
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=lr_end)

    log.info(f"Initialised final schedule with sigma_T = {sigma_T:.6f}")

    def get_window_mask():
        """All final-schedule levels + K-1 closest exploration levels below frontier."""
        window = torch.zeros(n_levels, dtype=torch.bool)
        # Final-schedule levels always stay in training
        for fs_sigma in final_schedule:
            fs_idx = int((all_sigmas - fs_sigma).abs().argmin().item())
            window[fs_idx] = True
        # K-1 closest exploration levels below the current frontier
        below = [i for i in range(n_levels)
                 if exploration_mask[i] and all_sigmas[i].item() < frontier]
        n_to_add = min(n_active_window - 1, len(below))
        for i in below[-n_to_add:]:
            window[i] = True
        return window

    # -----------------------------------------------------------------------
    round_idx = 0
    while True:
        # Stopping condition: epoch budget or round cap
        if total_epochs is not None and epochs_done >= total_epochs:
            log.info(f"Epoch budget ({total_epochs}) reached. Online algorithm complete.")
            break
        if total_epochs is None and round_idx >= max_rounds:
            log.info(f"Round cap ({max_rounds}) reached. Online algorithm complete.")
            break

        window_mask   = get_window_mask()
        active_sigmas = all_sigmas[window_mask]  # low→high window of K levels
        n_active      = active_sigmas.shape[0]

        budget_str = (
            f"epochs {epochs_done}/{total_epochs}" if total_epochs is not None
            else f"round {round_idx + 1}/{max_rounds}"
        )
        log.info(
            f"\n=== Round {round_idx + 1}  [{budget_str}]  |  "
            f"active window: {n_active}  |  "
            f"final schedule length: {len(final_schedule)}  |  "
            f"frontier: {frontier:.6f} ==="
        )

        # pass_frontier is fixed for the entire round; frontier is only updated after.
        pass_frontier  = frontier
        best_candidate = None  # best sigma found this round, committed at end

        # Effective epoch count grows with each round
        effective_epochs = epochs_per_round + round_idx * epoch_increase_per_round

        # Update model schedule for this round
        model.compute_schedule_variables(active_sigmas.to(device))

        def _eval_and_prune(label):
            """Evaluate B^2S from pass_frontier, prune intermediate levels, and
            track the best candidate.  Does NOT update frontier or final_schedule."""
            nonlocal best_candidate
            log.info(f"  -- {label}: B^2S from pass_frontier sigma={pass_frontier:.6f} --")
            cand_sigmas, ratios, cleans = compute_b_2step_from_frontier(
                model, val_loader, device, pass_frontier,
                n_batches=n_eval_batches, n_noise_samples=n_noise_samples,
            )
            sigma_next_val = None
            for sigma, ratio, clean in zip(cand_sigmas, ratios, cleans):
                sv = sigma.item()
                if np.isnan(ratio):
                    log.info(f"    sigma={sv:.5f}  B^2S=NaN  E_clean={clean:.3e}")
                    continue
                feasible = ratio <= tau
                log.info(
                    f"    sigma={sv:.5f}  B^2S={ratio:.4f}  "
                    f"E_2step={ratio * clean:.3e}  E_clean={clean:.3e}  "
                    f"{'FEASIBLE' if feasible else f'ratio={ratio:.3f}'}"
                )
                if feasible and sigma_next_val is None:
                    sigma_next_val = sv

            if sigma_next_val is None:
                return

            # Prune exploration levels strictly between sigma_next_val and pass_frontier.
            # sigma_next_val itself stays in exploration_mask so it remains in training.
            n_pruned = 0
            for idx in range(n_levels):
                if exploration_mask[idx]:
                    s = all_sigmas[idx].item()
                    if s > sigma_next_val and s < pass_frontier:
                        exploration_mask[idx] = False
                        n_pruned += 1
            log.info(
                f"  Pruned {n_pruned} exploration level(s) between "
                f"{sigma_next_val:.6f} and {pass_frontier:.6f}."
            )
            best_candidate = sigma_next_val
            # Recompute training window (frontier unchanged; best_candidate is now
            # the closest level below pass_frontier in the exploration mask)
            model.compute_schedule_variables(all_sigmas[get_window_mask()].to(device))

        # ---- Train for effective_epochs epochs, with optional mid-round evals ----
        epochs_this_round = 0
        for epoch in range(effective_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            epochs_this_round += 1
            current_lr = optimizer.param_groups[0]['lr']
            if (epoch + 1) % 10 == 0:
                log.info(
                    f"  epoch {epoch + 1}/{effective_epochs}  "
                    f"loss={loss:.6f}  lr={current_lr:.2e}"
                )

            budget_hit = (total_epochs is not None
                          and epochs_done + epochs_this_round >= total_epochs)
            is_last_epoch = (epoch + 1) == effective_epochs or budget_hit
            do_mid_eval   = (eval_every_epoch > 0
                             and (epoch + 1) % eval_every_epoch == 0
                             and not is_last_epoch)
            if do_mid_eval:
                _eval_and_prune(f"mid-round epoch {epoch + 1}")

            if budget_hit:
                log.info(
                    f"  Epoch budget ({total_epochs}) reached after epoch {epoch + 1} "
                    f"of round. Stopping round early."
                )
                break

        # ---- End-of-round eval ----
        _eval_and_prune("end-of-round")

        # ---- Per-level clean and inference errors over the current final schedule ----
        per_level_sigmas    = []
        per_level_clean     = []
        per_level_inference = []
        inf_bias_ratio      = float('nan')
        inf_clean_mse       = float('nan')
        if len(final_schedule) >= 1:
            fs_lohi = torch.tensor(sorted(final_schedule), dtype=torch.float32)
            per_level_sigmas, per_level_clean, per_level_inference = compute_per_level_errors(
                model, val_loader, device, fs_lohi, n_batches=n_eval_batches,
            )
            # Restore the training window schedule after eval
            model.compute_schedule_variables(all_sigmas[get_window_mask()].to(device))

            log.info(
                f"  Per-level errors (final schedule, {len(per_level_sigmas)} steps):"
            )
            for sigma, c_mse, i_mse in zip(per_level_sigmas, per_level_clean, per_level_inference):
                if not np.isnan(c_mse) and c_mse > 0:
                    ratio_str = f"{i_mse / c_mse:.4f}"
                else:
                    ratio_str = "N/A"
                log.info(
                    f"    sigma={sigma:.6f}  E_clean={c_mse:.3e}  "
                    f"E_inf={i_mse:.3e}  ratio={ratio_str}"
                )

            # Aggregate: frontier is the lowest level (index 0)
            if not np.isnan(per_level_clean[0]) and per_level_clean[0] > 0:
                inf_bias_ratio = per_level_inference[0] / per_level_clean[0]
                inf_clean_mse  = per_level_clean[0]
            log.info(
                f"  Inference bias (frontier): ratio={inf_bias_ratio:.4f}  "
                f"E_clean={inf_clean_mse:.3e}  "
                f"(schedule steps={len(final_schedule)})"
            )

        # ---- Commit best candidate to final schedule ----
        found_this_round = best_candidate is not None
        if found_this_round:
            log.info(
                f"  => Adding sigma_next={best_candidate:.6f} to final schedule  "
                f"(jump from frontier {pass_frontier:.6f})"
            )
            final_schedule.append(best_candidate)
            frontier = best_candidate

        if found_this_round:
            patience_counter = 0
        else:
            patience_counter += 1
            log.info(
                f"  No new level found this round "
                f"({patience_counter}/{patience_limit})."
            )

        epochs_done += epochs_this_round
        round_idx   += 1

        # Save model checkpoint and state after every round
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_round_{round_idx}.pth')
        torch.save(model.state_dict(), ckpt_path)

        state = {
            'round'               : round_idx,
            'epochs_done'         : epochs_done,
            'final_schedule'      : final_schedule,
            'frontier'            : frontier,
            'n_active'            : int(exploration_mask.sum().item()),
            'patience_counter'    : patience_counter,
            'last_checkpoint'     : ckpt_path,
            'inference_bias'      : inf_bias_ratio,
            'inference_clean'     : inf_clean_mse,
            'per_level_sigmas'    : per_level_sigmas,
            'per_level_clean_mse' : per_level_clean,
            'per_level_inf_mse'   : per_level_inference,
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        if patience_counter >= patience_limit:
            log.info("Patience exhausted. Online algorithm complete.")
            break

    # -----------------------------------------------------------------------
    # Save final schedule in schedule.json format (compatible with eval scripts)
    # final_schedule is ordered high → low; reverse to low → high for output
    schedule_lo_hi = list(reversed(final_schedule))

    output = {
        'tau'           : tau,
        'n_eval_batches': n_eval_batches,
        'schedule'      : schedule_lo_hi,
        'schedule_log10': [float(np.log10(s)) for s in schedule_lo_hi],
        'n_steps'       : len(schedule_lo_hi),
        'sigma_min'     : schedule_lo_hi[0],
        'sigma_max'     : schedule_lo_hi[-1],
        'algorithm'     : 'online',
    }
    schedule_path = os.path.join(checkpoint_dir, 'schedule.json')
    with open(schedule_path, 'w') as f:
        json.dump(output, f, indent=2)

    log.info(f"\nOnline algorithm finished. Final schedule ({len(schedule_lo_hi)} steps):")
    for i, s in enumerate(schedule_lo_hi):
        log.info(f"  t={i}: sigma={s:.6f}  (log10={np.log10(s):.3f})")
    log.info(f"Schedule saved to {schedule_path}")
    log.info(f"State saved to {state_path}")


if __name__ == '__main__':
    main()
