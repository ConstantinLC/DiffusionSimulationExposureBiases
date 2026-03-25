#!/usr/bin/env python
"""
Evaluate the Wiener-PSD prediction for the own-prediction bias.

First-order Taylor expansion of B^{own} (squaring eps + Jd):

    B^{own}_pred = 1 + (2*alpha/sigma^2) * R2 + (alpha/sigma^2)^2 * R3

where R2 = sum_k E(k)^2 / sum_k E(k)  (cross-term: 2<eps, Jd>/||eps||^2)
      R3 = sum_k E(k)^3 / sum_k E(k)  (quadratic term: ||Jd||^2/||eps||^2)
      E(k) = |FFT(epsilon_clean)_k|^2 / N is the per-mode error PSD,
      N = number of spatial pixels, and the sum runs over all Fourier modes.

This is compared to the empirical B^{own} = mean(E_own / E_clean) at each
noise level.  Both quantities are computed from the validation set.

Usage:
  python experiments/eval_wiener_prediction.py \
      --checkpoint_dir checkpoints/KolmogorovFlow/forecasting/DiffusionModel_inverseCosLog-1.875_20_12 \
      --checkpoint_name best_model.pth \
      --output results/wiener_prediction.pdf
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.func import jvp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig
from src.data.loaders import get_data_loaders
from src.models.diffusion import DiffusionModel

_DATA_CONFIG_FIELDS = {
    'dataset_name', 'data_path', 'resolution', 'prediction_steps',
    'frames_per_step', 'traj_length', 'frames_per_time_step',
    'limit_trajectories_train', 'limit_trajectories_val',
    'super_resolution', 'batch_size', 'val_batch_size',
}

plt.rcParams.update({
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


def wiener_bias_prediction(eps_clean: torch.Tensor, alpha: float, sigma: float) -> float:
    """
    Compute the first-order Taylor prediction of B^{own}:

        B^{own}_pred = 1 + (2*alpha/sigma^2) * R2 + (alpha/sigma^2)^2 * R3

    where R2 = sum_k E(k)^2 / sum_k E(k)   (cross-term: 2<eps, Jd>/||eps||^2)
          R3 = sum_k E(k)^3 / sum_k E(k)   (quadratic term: ||Jd||^2/||eps||^2)

    and E(k) = P_k / N,  P_k = sum_c |FFT(eps_clean_c)_k|^2,  N = H*W (or L).

    eps_clean : (C, H, W) or (C, L)  — a single sample (no batch dim)
    """
    if eps_clean.ndim == 3:          # 2D: (C, H, W)
        X = torch.fft.fft2(eps_clean, dim=(-2, -1))
        N = eps_clean.shape[-2] * eps_clean.shape[-1]
    else:                             # 1D: (C, L)
        X = torch.fft.fft(eps_clean, dim=-1)
        N = eps_clean.shape[-1]

    P_k     = X.abs().pow(2).sum(dim=0)   # sum over channels -> (H,W) or (L,)
    P_total = P_k.sum()
    if P_total < 1e-30:
        return float('nan')

    coeff = alpha / sigma**2
    # R2 = (1/N) * sum_k P_k^2 / sum_k P_k  (maps to sum_k E(k)^2 / sum_k E(k))
    R2 = (P_k.pow(2).sum() / P_total / N).item()
    # R3 = (1/N^2) * sum_k P_k^3 / sum_k P_k  (maps to sum_k E(k)^3 / sum_k E(k))
    R3 = (P_k.pow(3).sum() / P_total / N**2).item()
    return 1.0 + 2.0 * coeff * R2 + coeff**2 * R3


def make_denoiser_fn(model, cond, t):
    """Pure function f(d_noisy) -> x0_estimate for a single sample (no batch dim)."""
    sqrt_alpha = model.sqrtAlphasCumprod[t[0]]
    sqrt_one_minus_alpha = model.sqrtOneMinusAlphasCumprod[t[0]]
    C_cond = cond.shape[1]

    def denoiser(d_noisy):
        d_noisy_b = d_noisy.unsqueeze(0)
        cond_input = torch.cat((cond, d_noisy_b), dim=1)
        pred_noise = model.unet(cond_input, t)[:, C_cond:]
        x0 = (d_noisy_b - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
        return x0.squeeze(0)

    return denoiser


@torch.no_grad()
def evaluate(model, val_loader, device, n_batches=20):
    """
    For each noise level, compute:
      - bias_empirical : mean(E_own / E_clean) over samples
      - bias_predicted : eq. (69), averaged over samples
      - mse_clean      : mean E_clean
    Returns dict with arrays indexed by noise level (0 = highest sigma).
    """
    model.eval()
    T      = model.timesteps
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()
    alphas = model.sqrtAlphasCumprod.squeeze().cpu() ** 2   # bar{alpha}

    bias_emp_acc  = np.zeros(T)
    mse_clean_acc = np.zeros(T)
    # Accumulate P_k over ALL samples before computing the ratio.
    # This gives a consistent estimator of E(k) = ensemble-average PSD,
    # avoiding the upward bias from E[P_k^2] > E(k)^2 in per-sample estimates.
    psd_acc   = [None] * T   # will hold sum of P_k tensors
    psd_count = np.zeros(T)
    count = 0

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        data   = sample['data'].to(device)
        cond   = data[:, 0]
        target = data[:, 1]
        spatial_dims = tuple(range(1, target.ndim))

        _, x0_clean = model(
            conditioning=cond, data=target,
            return_x0_estimate=True, input_type='clean',
        )
        _, x0_own = model(
            conditioning=cond, data=target,
            return_x0_estimate=True, input_type='own-pred',
        )

        for t_idx in range(T):
            # x0_clean is built inside `for i in reversed(range(T))`, so
            # list index 0 = estimate at timestep T-1 (highest sigma).
            # sigma index t_idx corresponds to list index T-1-t_idx.
            est_idx = T - 1 - t_idx
            eps_clean_batch = x0_clean[est_idx] - target   # (B, C, ...)
            eps_own_batch   = x0_own[est_idx]   - target

            mse_clean = eps_clean_batch.pow(2).mean(dim=spatial_dims)  # (B,)
            mse_own   = eps_own_batch.pow(2).mean(dim=spatial_dims)

            valid = mse_clean > 1e-12
            if valid.any():
                bias_emp_acc[t_idx] += (mse_own[valid] / mse_clean[valid]).mean().item()
            mse_clean_acc[t_idx] += mse_clean.mean().item()

            # Accumulate P_k = sum_c |FFT(eps_clean_c)|^2 over all samples
            if eps_clean_batch.ndim == 4:   # 2D: (B, C, H, W)
                X = torch.fft.fft2(eps_clean_batch, dim=(-2, -1))
            else:                            # 1D: (B, C, L)
                X = torch.fft.fft(eps_clean_batch, dim=-1)
            P_k = X.abs().pow(2).sum(dim=1).mean(dim=0)  # sum channels, mean batch -> spatial shape

            if psd_acc[t_idx] is None:
                psd_acc[t_idx] = P_k.detach().cpu()
            else:
                psd_acc[t_idx] += P_k.detach().cpu()
            psd_count[t_idx] += 1

        count += 1

    # Compute Wiener prediction from the ensemble-averaged PSD
    bias_pred = np.zeros(T)
    for t_idx in range(T):
        if psd_acc[t_idx] is None or psd_count[t_idx] == 0:
            bias_pred[t_idx] = float('nan')
            continue
        P_k_mean = psd_acc[t_idx] / psd_count[t_idx]   # ensemble-averaged P_k
        sigma_t  = float(sigmas[t_idx])
        alpha_t  = float(alphas[t_idx])
        N        = P_k_mean.numel()
        P_total  = P_k_mean.sum()
        coeff    = alpha_t / sigma_t**2
        R2       = (P_k_mean.pow(2).sum() / P_total / N).item()
        R3       = (P_k_mean.pow(3).sum() / P_total / N**2).item()
        bias_pred[t_idx] = 1.0 + 2.0 * coeff * R2 + coeff**2 * R3

    return {
        'sigmas':         sigmas.numpy(),
        'alphas':         alphas.numpy(),
        'bias_empirical': bias_emp_acc / max(count, 1),
        'bias_predicted': bias_pred,
        'mse_clean':      mse_clean_acc / max(count, 1),
    }


def evaluate_taylor(model, val_loader, device, n_batches=5):
    """
    Compute B^{own} decomposition via JVP + remainder.

    Full decomposition (exact up to Taylor remainder):
      eps_own = eps_clean + Jd + R     where d = sqrt(alpha)*eps_clean
      B = ||eps_own||^2 / ||eps||^2
        = 1 + [2<eps,Jd> + ||Jd||^2] / ||eps||^2        (Taylor)
            + [||R||^2 + 2<eps,R> + 2<Jd,R>] / ||eps||^2 (remainder)

    Returns dict with 'bias_taylor' (1st-order only) and 'bias_taylor_full'
    (including remainder, should match empirical exactly).
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    T = model.timesteps
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze()
    sqrt_alphas = model.sqrtAlphasCumprod.squeeze()

    # Taylor terms (normalised by ||eps||^2)
    cross_acc = np.zeros(T)       # 2<eps, Jd>
    quad_acc = np.zeros(T)        # ||Jd||^2
    # Remainder terms
    R_sq_acc = np.zeros(T)        # ||R||^2
    cross_eps_R_acc = np.zeros(T) # 2<eps, R>
    cross_Jd_R_acc = np.zeros(T)  # 2<Jd, R>
    count = np.zeros(T)

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        data = sample['data'].to(device)
        cond = data[:1, 0]        # first sample only
        target = data[:1, 1]
        y = target.squeeze(0)     # (C, *spatial)

        for t_idx in range(T):
            i = T - 1 - t_idx     # actual timestep index
            t = torch.full((1,), i, device=device, dtype=torch.long)

            sqrt_alpha_t = sqrt_alphas[i]
            sigma_t = sigmas[i]

            noise = torch.randn_like(y)
            y_noisy = sqrt_alpha_t * y + sigma_t * noise

            denoiser_fn = make_denoiser_fn(model, cond, t)

            # u_clean = denoiser(y_noisy)
            with torch.enable_grad():
                u_clean = denoiser_fn(y_noisy)

            eps_clean = u_clean - y
            eps_sq = (eps_clean * eps_clean).sum().item()
            if eps_sq < 1e-24:
                continue

            # Jd = J * delta via JVP, where delta = sqrt(alpha) * eps_clean
            delta = sqrt_alpha_t * eps_clean
            with torch.enable_grad():
                _, Jd = jvp(denoiser_fn, (y_noisy,), (delta,))

            # u_own = denoiser(y_noisy + delta) — actual own-prediction output
            with torch.no_grad():
                u_own = denoiser_fn(y_noisy + delta)

            # R = (u_own - y) - eps_clean - Jd  (Taylor remainder)
            R = (u_own - y) - eps_clean - Jd

            cross_acc[t_idx] += 2.0 * (eps_clean * Jd).sum().item() / eps_sq
            quad_acc[t_idx] += (Jd * Jd).sum().item() / eps_sq
            R_sq_acc[t_idx] += (R * R).sum().item() / eps_sq
            cross_eps_R_acc[t_idx] += 2.0 * (eps_clean * R).sum().item() / eps_sq
            cross_Jd_R_acc[t_idx] += 2.0 * (Jd * R).sum().item() / eps_sq
            count[t_idx] += 1

        print(f"  Taylor eval: batch {batch_idx+1}/{n_batches}")

    c = np.maximum(count, 1)
    valid = count > 0
    taylor_terms = (cross_acc + quad_acc) / c
    remainder_terms = (R_sq_acc + cross_eps_R_acc + cross_Jd_R_acc) / c

    return {
        'bias_taylor': np.where(valid, 1.0 + taylor_terms, np.nan),
        'bias_taylor_full': np.where(valid, 1.0 + taylor_terms + remainder_terms, np.nan),
    }


def plot(results, output_path):
    sigmas      = results['sigmas']
    b_emp       = results['bias_empirical']
    b_wiener    = results['bias_predicted']
    b_taylor    = results.get('bias_taylor')
    b_full      = results.get('bias_taylor_full')
    mse         = results['mse_clean']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: B^own vs sigma ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(sigmas, b_emp,    'o-',  label=r'Empirical $\mathcal{B}^{own}$', color='C0')
    ax.plot(sigmas, b_wiener, 's--', label=r'Wiener PSD',                    color='C1')
    if b_taylor is not None:
        ax.plot(sigmas, b_taylor, '^--', label=r'Taylor $1^{st}$ order (JVP)', color='C2')
    if b_full is not None:
        ax.plot(sigmas, b_full,   'D--', label=r'Taylor + remainder',          color='C3')
    ax.axhline(1.0, color='gray', lw=1, ls=':')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\mathcal{B}^{own}$')
    ax.set_title(r'Own-prediction bias vs $\sigma$')
    ax.legend(fontsize=10)

    # ── Right: scatter predicted vs empirical ────────────────────────────────
    ax = axes[1]
    all_vals = [b_emp, b_wiener]
    if b_taylor is not None:
        all_vals.append(b_taylor[~np.isnan(b_taylor)])
    if b_full is not None:
        all_vals.append(b_full[~np.isnan(b_full)])
    all_vals = np.concatenate(all_vals)
    lims = [all_vals.min() * 0.98, all_vals.max() * 1.02]
    ax.plot(lims, lims, 'k--', lw=1, label='y = x')
    ax.scatter(b_wiener, b_emp, c=sigmas, cmap='viridis', s=60, zorder=3,
               marker='s', label='Wiener')
    if b_taylor is not None:
        ax.scatter(b_taylor, b_emp, c=sigmas, cmap='viridis', s=60, zorder=3,
                   marker='^', label='Taylor 1st order')
    if b_full is not None:
        sc = ax.scatter(b_full, b_emp, c=sigmas, cmap='viridis', s=60, zorder=3,
                        marker='D', label='Taylor + remainder')
    else:
        sc = ax.scatter(b_wiener, b_emp, c=sigmas, cmap='viridis', s=60, zorder=3)
    plt.colorbar(sc, ax=ax, label=r'$\sigma$')
    ax.set_xlabel(r'Predicted $\mathcal{B}^{own}$')
    ax.set_ylabel(r'Empirical $\mathcal{B}^{own}$')
    ax.set_title('Predicted vs empirical')
    ax.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved → {output_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    header = f"{'sigma':>8}  {'E_clean':>12}  {'B_emp':>8}  {'B_wiener':>8}"
    if b_taylor is not None:
        header += f"  {'B_taylor':>8}"
    if b_full is not None:
        header += f"  {'B_full':>8}"
    print(f"\n{header}")
    for i in range(len(sigmas)):
        line = f"{sigmas[i]:>8.4f}  {mse[i]:>12.4e}  {b_emp[i]:>8.4f}  {b_wiener[i]:>8.4f}"
        if b_taylor is not None:
            line += f"  {b_taylor[i]:>8.4f}"
        if b_full is not None:
            line += f"  {b_full[i]:>8.4f}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir',  required=True)
    parser.add_argument('--checkpoint_name', default='best_model.pth')
    parser.add_argument('--output',    default='results/wiener_prediction.pdf')
    parser.add_argument('--n_batches', type=int, default=20)
    parser.add_argument('--n_batches_taylor', type=int, default=5,
                        help='Batches for JVP-based Taylor prediction (expensive)')
    parser.add_argument('--skip_taylor', action='store_true',
                        help='Skip JVP-based Taylor prediction')
    parser.add_argument('--device',    default='cuda')
    args = parser.parse_args()

    cfg_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    data_params  = cfg['data_params']
    model_params = cfg['model_params']

    data_cfg = DataConfig(**{k: v for k, v in data_params.items() if k in _DATA_CONFIG_FIELDS})
    _, val_loader, _ = get_data_loaders(data_cfg)

    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model = DiffusionModel(
        checkpoint=ckpt_path,
        load_betas=True,
        dimension=model_params['dimension'],
        dataSize=model_params['dataSize'],
        condChannels=model_params['condChannels'],
        dataChannels=model_params['dataChannels'],
        diffSchedule=model_params['diffSchedule'],
        diffSteps=model_params['diffSteps'],
        inferenceSamplingMode=model_params['inferenceSamplingMode'],
        inferenceConditioningIntegration=model_params['inferenceConditioningIntegration'],
        diffCondIntegration=model_params['diffCondIntegration'],
        padding_mode=model_params.get('padding_mode', 'circular'),
        architecture=model_params.get('architecture', 'ours'),
    ).to(args.device)
    model.eval()
    model.compute_schedule_variables(sigmas=model.sqrtOneMinusAlphasCumprod.ravel()[-20:])
    print(f"Loaded: {ckpt_path}")
    print(f"Schedule: T={model.timesteps}, "
          f"sigma in [{model.sqrtOneMinusAlphasCumprod.min():.3f}, "
          f"{model.sqrtOneMinusAlphasCumprod.max():.3f}]")

    print("\n=== Evaluating Wiener PSD prediction ===")
    results = evaluate(model, val_loader, args.device, n_batches=args.n_batches)

    if not args.skip_taylor:
        print("\n=== Evaluating Taylor prediction (JVP + remainder) ===")
        taylor_results = evaluate_taylor(
            model, val_loader, args.device, n_batches=args.n_batches_taylor,
        )
        results['bias_taylor'] = taylor_results['bias_taylor']
        results['bias_taylor_full'] = taylor_results['bias_taylor_full']

    plot(results, args.output)


if __name__ == '__main__':
    main()
