#!/usr/bin/env python
"""
Cross-model "prev-pred" evaluation — two curves:

  1. Adjacent cross-model (2 forward passes):
       target + noise(sigma_coarse)   → model_coarse single step → x0_coarse
       x0_coarse + noise(sigma_fine)  → model_fine   single step → x0

  2. Full cascade / ancestor sampling (N forward passes, one per sigma level):
       target + noise(sigma_max)  → model[max] → x0_max
       x0_max + noise(sigma_k-1)  → model[k-1] → x0_k-1
       ...
       x0_{k+1} + noise(sigma_k)  → model[k]   → x0_k    ← plotted at sigma_k

  Both are compared against the same-model metrics from eval_single_sigma_sweep.

Usage:
  python experiments/eval_cross_sigma_prev_pred.py \\
      --checkpoint_dir ./checkpoints/KuramotoSivashinsky/single_sigma \\
      --output results/cross_sigma_prev_pred.pdf
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from experiments.eval_dw_bias import build_model, evaluate_dw_bias

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
}


# ── Directory helpers ──────────────────────────────────────────────────────────

def find_run_dir(sigma_dir: str) -> str | None:
    runs = [d for d in os.listdir(sigma_dir) if d.startswith('PDERefiner')]
    if not runs:
        return None
    candidate = os.path.join(sigma_dir, runs[0])
    return candidate if os.path.exists(os.path.join(candidate, 'best_model.pth')) else None


def collect_sigma_dirs(checkpoint_dir: str) -> list[tuple[float, str]]:
    """Return (log_sigma, run_dir) sorted ascending (finest sigma first)."""
    entries = []
    for name in os.listdir(checkpoint_dir):
        if not name.startswith('sigma_'):
            continue
        log_sigma = float(name[len('sigma_'):])
        run_dir = find_run_dir(os.path.join(checkpoint_dir, name))
        if run_dir is None:
            print(f"  [skip] {name}: no completed run found")
            continue
        entries.append((log_sigma, run_dir))
    return sorted(entries, key=lambda x: x[0])


def load_model(run_dir: str, device: str):
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    model = build_model(cfg['model_params']).to(device)
    model.load_state_dict(
        torch.load(os.path.join(run_dir, 'best_model.pth'), map_location=device)
    )
    return model


# ── Evaluation functions ───────────────────────────────────────────────────────

def _single_step(model, cond, x0_ancestor, t0, device):
    """
    One denoising step of `model` at its sigma_min level (t=0),
    starting from `x0_ancestor` re-noised at that level.
    Returns x0 estimate.
    """
    sigma = model.sigmas[t0]
    noisy = x0_ancestor + sigma * torch.randn_like(x0_ancestor)
    inp   = torch.cat((cond, noisy), dim=1)
    eps   = model._apply_unet(inp, t0, step_idx=0)[:, cond.shape[1]:]
    return noisy - sigma * eps


def evaluate_cross_model(model_fine, model_coarse, val_loader, device) -> float:
    """
    Adjacent cross-model prev-pred (2 forward passes total):
      target + noise(sigma_coarse) → model_coarse → x0_coarse
      x0_coarse + noise(sigma_fine) → model_fine  → x0_final
    Returns mean MSE(x0_final, target) over the val set.
    """
    model_fine.eval()
    model_coarse.eval()

    mse_list = []
    with torch.no_grad():
        for sample in val_loader:
            data   = sample["data"].to(device)
            cond   = data[:, 0]
            target = data[:, 1]
            t0 = torch.zeros(target.shape[0], device=device, dtype=torch.long)

            x0_coarse = _single_step(model_coarse, cond, target,   t0, device)
            x0_final  = _single_step(model_fine,   cond, x0_coarse, t0, device)

            mse_list.append(torch.mean((x0_final - target) ** 2).item())

    return float(np.mean(mse_list))


def evaluate_cascade_single(models_coarse_to_fine, val_loader, device) -> float:
    """
    Run a cascade of exactly len(models_coarse_to_fine) steps and return the
    final MSE.  Reuses _single_step; models must be in coarse-to-fine order.
    """
    for m in models_coarse_to_fine:
        m.eval()

    mse_list = []
    with torch.no_grad():
        for sample in val_loader:
            data   = sample["data"].to(device)
            cond   = data[:, 0]
            target = data[:, 1]
            t0 = torch.zeros(target.shape[0], device=device, dtype=torch.long)

            x0 = target
            for model in models_coarse_to_fine:
                x0 = _single_step(model, cond, x0, t0, device)

            mse_list.append(torch.mean((x0 - target) ** 2).item())

    return float(np.mean(mse_list))


def evaluate_full_cascade(models_coarse_to_fine, val_loader, device) -> list[float]:
    """
    Full ancestor-sampling cascade across all sigma levels.

    models_coarse_to_fine: list of models ordered from coarsest (highest sigma)
                           to finest (lowest sigma).

    For each batch:
      - Start: target + noise(sigma_max) → model[0] → x0[0]
      - Step k: x0[k-1] + noise(sigma_k) → model[k] → x0[k]

    Returns a list of mean MSE values (one per model, in coarse-to-fine order),
    where MSE[k] is the error after the cascade has reached model[k].
    """
    for m in models_coarse_to_fine:
        m.eval()

    n = len(models_coarse_to_fine)
    mse_lists = [[] for _ in range(n)]

    with torch.no_grad():
        for sample in val_loader:
            data   = sample["data"].to(device)
            cond   = data[:, 0]
            target = data[:, 1]
            t0 = torch.zeros(target.shape[0], device=device, dtype=torch.long)

            x0 = target  # clean input for the first (coarsest) step
            for k, model in enumerate(models_coarse_to_fine):
                x0 = _single_step(model, cond, x0, t0, device)
                mse_lists[k].append(torch.mean((x0 - target) ** 2).item())

    return [float(np.mean(lst)) for lst in mse_lists]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir',
                        default='./checkpoints/KuramotoSivashinsky/single_sigma',
                        help='Root directory of the sigma sweep')
    parser.add_argument('--output', default='results/cross_sigma_prev_pred.pdf')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Discover runs (sorted ascending: entries[0]=finest, entries[-1]=coarsest) ──
    entries = collect_sigma_dirs(args.checkpoint_dir)
    if len(entries) < 2:
        print("Need at least 2 sigma levels. Exiting.")
        return
    print(f"Found {len(entries)} sigma runs: {[e[0] for e in entries]}")

    # ── Build val_loader ───────────────────────────────────────────────────────
    with open(os.path.join(entries[0][1], 'config.json')) as f:
        first_cfg = json.load(f)
    data_cfg = DataConfig(**{
        **{k: v for k, v in first_cfg['data_params'].items() if k in _DATA_CONFIG_FIELDS},
        'batch_size':     args.batch_size,
        'val_batch_size': args.batch_size,
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    # ── Load all models upfront ────────────────────────────────────────────────
    print("\nLoading models...")
    all_models = [load_model(run_dir, args.device) for _, run_dir in entries]
    # all_models[0] = finest model, all_models[-1] = coarsest model

    log_sigmas = np.array([e[0] for e in entries])   # ascending (finest first)

    # ── Adjacent cross-model evaluation ───────────────────────────────────────
    # For each fine model entries[i], use entries[i+1] (coarser) as the prev-pred.
    # entries[-1] (coarsest) is skipped — it has no coarser neighbour.
    print("\nEvaluating adjacent cross-model pairs...")
    mse_cross    = []
    mse_ancestor = []
    mse_own_pred = []
    mse_clean    = []

    for i in range(len(entries) - 1):
        log_sigma_fine, _ = entries[i]
        print(f"  fine=10^{log_sigma_fine:.2f}  coarse=10^{entries[i+1][0]:.2f}")

        mse_cr = evaluate_cross_model(all_models[i], all_models[i + 1], val_loader, args.device)

        results  = evaluate_dw_bias(all_models[i], val_loader, args.device)
        mse_anc  = results['mse_ancestor'][-1]
        mse_own  = results['mse_clean_own_pred'][-1]
        mse_cln  = results['mse_clean'][-1]

        print(f"    cross-model: {mse_cr:.4e}  |  ancestor: {mse_anc:.4e}"
              f"  |  own-pred: {mse_own:.4e}  |  clean: {mse_cln:.4e}")

        mse_cross.append(mse_cr)
        mse_ancestor.append(mse_anc)
        mse_own_pred.append(mse_own)
        mse_clean.append(mse_cln)

    # xs for adjacent-pair curves: all sigma values except the coarsest
    xs_pairs = log_sigmas[:-1]

    # ── N=3 cascade evaluation ─────────────────────────────────────────────────
    # For each fine model at index i, use the window [i+2, i+1, i] (coarse→fine).
    # The 2 coarsest entries are skipped (no 2 coarser neighbours available).
    print("\nEvaluating N=3 cascade...")
    mse_n3 = []
    for i in range(len(entries) - 2):
        window_c2f = [all_models[i + 2], all_models[i + 1], all_models[i]]
        mse = evaluate_cascade_single(window_c2f, val_loader, args.device)
        print(f"  fine=10^{entries[i][0]:.2f}  N=3 cascade MSE: {mse:.4e}")
        mse_n3.append(mse)

    xs_n3 = log_sigmas[:-2]   # all except the 2 coarsest

    # ── Full cascade evaluation ────────────────────────────────────────────────
    # Models ordered coarse-to-fine = reversed all_models list
    print("\nEvaluating full cascade (ancestor sampling across all sigma levels)...")
    models_c2f = list(reversed(all_models))
    cascade_mses_c2f = evaluate_full_cascade(models_c2f, val_loader, args.device)

    # Align with log_sigmas (ascending = finest first): reverse the coarse-to-fine list
    mse_cascade = list(reversed(cascade_mses_c2f))   # now indexed finest-first
    xs_cascade  = log_sigmas                          # all sigma values

    for ls, mc in zip(xs_cascade, mse_cascade):
        print(f"  sigma=10^{ls:.2f}  cascade MSE: {mc:.4e}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(xs_pairs,   mse_cross,    'D-',  color='red',
            label='Cross-model cascade N=2')
    #ax.plot(xs_n3,      mse_n3,       'h-',  color='magenta',
    #        label='Cross-model cascade N=3')
    ax.plot(xs_cascade, mse_cascade,  'P-',  color='purple',
            label=f'Full cascade N={len(entries)}')
    #ax.plot(xs_pairs,   mse_ancestor, '^-',  color='orange',
    #        label='Same-model prev-pred (cold-start)')
    ax.plot(xs_pairs,   mse_own_pred, 's-',  color='green',
            label='Same-model own-pred')
    ax.plot(xs_pairs,   mse_clean,    'o--', color='blue',
            label='Clean input (train distribution)', alpha=0.6)

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(\sigma_\mathrm{fine})$', fontsize=12)
    ax.set_ylabel('MSE vs ground truth', fontsize=12)
    ax.set_title('Cross-model cascade vs same-model inference')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"\nPlot saved to {args.output}")
    plt.close(fig)


if __name__ == '__main__':
    main()
