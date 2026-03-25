#!/usr/bin/env python
"""
Test 3: Empirical Rayleigh quotient lambda_e vs Wiener prediction.

For each checkpoint and noise level sigma, compute:

  lambda_e^{empirical} = e^T J_theta e   (via JVP in direction of epsilon_clean)

  lambda_e^{Wiener}    = (sqrt(alpha) / sigma^2) * sum_k |e_hat_k|^2 * E(k)
                       = (sqrt(alpha) / sigma^2) * <|e_hat|^2, E>

where e = epsilon_clean / ||epsilon_clean|| and E(k) is the empirical error PSD.

If lambda_e^{empirical} << lambda_e^{Wiener} and the gap grows with training,
the model has genuinely near-zero Jacobian in its residual error direction
(near-collapse), not just a spectral shift of errors.

Plots:
  Left:  lambda_e^{empirical} and lambda_e^{Wiener} vs sigma, per checkpoint
  Right: ratio lambda_e^{empirical} / lambda_e^{Wiener} vs sigma, per checkpoint
         (ratio < 1 means near-collapse; ratio shrinking with training confirms it)

Usage:
  python experiments/eval_jacobian_vs_wiener.py \
      --checkpoint_dir checkpoints/KolmogorovFlow/forecasting/DiffusionModel_inverseCosLog-1.875_20_12 \
      --checkpoint_names epoch_501.pth epoch_1001.pth best_model.pth \
      --output results/jacobian_vs_wiener.pdf \
      --n_batches 10 --T 10
"""
import os
import sys
import argparse
import json

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
    'font.size': 13,
    'axes.linewidth': 1.4,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_denoiser_fn(model, cond, t):
    """Pure function f(d_noisy) -> x0_estimate for a single sample (no batch dim)."""
    sqrt_alpha         = model.sqrtAlphasCumprod[t[0]]
    sqrt_one_minus_alpha = model.sqrtOneMinusAlphasCumprod[t[0]]
    C_cond = cond.shape[1]

    def denoiser(d_noisy):
        d_noisy_b  = d_noisy.unsqueeze(0)
        cond_input = torch.cat((cond, d_noisy_b), dim=1)
        pred_noise = model.unet(cond_input, t)[:, C_cond:]
        x0 = (d_noisy_b - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
        return x0.squeeze(0)

    return denoiser


def wiener_lambda(eps_clean, sqrt_alpha, sigma):
    """
    Compute lambda_e^{Wiener} = (sqrt_alpha / sigma^2) * (1/N) * sum_k P_k^2 / sum_k P_k

    where P_k = sum_c |FFT(eps_clean_c)_k|^2  and  N = number of spatial pixels.

    Derivation: for an optimal Wiener denoiser, the input Jacobian eigenvalue at
    mode k is lambda_k = (sqrt_alpha/sigma^2) * E_k, where E_k is the per-mode
    error variance. Approximating E_k ~ P_k/N from the single-sample error, and
    weighting by the spectral power w_k = P_k/sum_k P_k gives this formula.

    Correctly normalised — avoids the O((HW)^2) inflation from raw FFT amplitudes.
    Works for 2D (C,H,W) and 1D (C,L) fields.
    """
    if eps_clean.ndim == 3:   # 2D: (C, H, W)
        _, H, W = eps_clean.shape
        X = torch.fft.fft2(eps_clean, dim=(-2, -1))
        N = H * W
    else:                      # 1D: (C, L)
        X = torch.fft.fft(eps_clean, dim=-1)
        N = eps_clean.shape[-1]

    P_k     = X.abs().pow(2).sum(dim=0)   # sum over channels
    P_total = P_k.sum()
    if P_total < 1e-30:
        return float('nan')

    return float(sqrt_alpha / sigma ** 2 / N * (P_k.pow(2).sum() / P_total))


# ── Main collection ───────────────────────────────────────────────────────────

def collect(model, val_loader, device, n_batches=10, T=10):
    """
    Returns (sigmas, lambda_emp, lambda_wiener) each of length T.
    JVP is run per sample; results are averaged over samples.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sigmas_all   = model.sqrtOneMinusAlphasCumprod.squeeze()
    sqrt_alphas  = model.sqrtAlphasCumprod.squeeze()
    T_full       = len(sigmas_all)

    indices = torch.linspace(T_full - T, T_full - 1, T).long().flip(0)
    sigmas  = sigmas_all[indices].cpu()

    lambda_emp_acc    = np.zeros(T)
    lambda_wiener_acc = np.zeros(T)
    count             = np.zeros(T)

    is_2d = None

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        data   = sample['data'].to(device)
        cond_batch  = data[:, 0]
        target_batch = data[:, 1]

        if is_2d is None:
            is_2d = (target_batch.ndim == 4)

        # Process first sample in the batch (JVP is per-sample)
        cond   = cond_batch[:1]
        target = target_batch[:1]
        y      = target.squeeze(0)   # (C, *spatial)

        for local_t, global_t in enumerate(indices.tolist()):
            sigma_t      = sigmas_all[global_t]
            sqrt_alpha_t = sqrt_alphas[global_t]

            t = torch.full((1,), global_t, device=device, dtype=torch.long)

            noise  = torch.randn_like(y)
            y_noisy = sqrt_alpha_t * y + sigma_t * noise

            denoiser_fn = make_denoiser_fn(model, cond, t)

            with torch.enable_grad():
                u_clean = denoiser_fn(y_noisy)

            eps_clean = u_clean - y          # (C, *spatial)
            eps_norm  = eps_clean.norm()
            if eps_norm < 1e-12:
                continue
            e_hat = eps_clean / eps_norm     # unit-norm error direction

            # ── Empirical Rayleigh quotient: e^T J e via JVP ─────────────────
            # tangent = sqrt_alpha * e_hat  (the actual perturbation direction)
            tangent = sqrt_alpha_t * e_hat
            with torch.enable_grad():
                _, J_tangent = jvp(denoiser_fn, (y_noisy,), (tangent,))
            # lambda_e = <e_hat, J_theta * sqrt_alpha * e_hat> / sqrt_alpha
            #          = <e_hat, J_tangent> / sqrt_alpha
            lambda_e_emp = (e_hat * J_tangent).sum().item() / float(sqrt_alpha_t)

            # ── Wiener prediction ─────────────────────────────────────────────
            lambda_e_wiener = wiener_lambda(
                eps_clean.detach(), float(sqrt_alpha_t), float(sigma_t)
            )

            lambda_emp_acc[local_t]    += lambda_e_emp
            lambda_wiener_acc[local_t] += lambda_e_wiener
            count[local_t]             += 1

    valid = count > 0
    lambda_emp    = np.where(valid, lambda_emp_acc    / np.maximum(count, 1), np.nan)
    lambda_wiener = np.where(valid, lambda_wiener_acc / np.maximum(count, 1), np.nan)

    return sigmas.numpy(), lambda_emp, lambda_wiener


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(all_results, output_path):
    """
    all_results: list of (label, sigmas, lambda_emp, lambda_wiener)
    """
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(all_results)))
    fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(12, 4.5))

    for (label, sigmas, l_emp, l_wiener), color in zip(all_results, colors):
        ax_abs.plot(sigmas, l_emp,    color=color, ls='-',  label=f'{label} (empirical)')
        ax_abs.plot(sigmas, l_wiener, color=color, ls='--', label=f'{label} (Wiener)')

        ratio = np.where(np.abs(l_wiener) > 1e-30, l_emp / l_wiener, np.nan)
        ax_ratio.plot(sigmas, ratio, color=color, label=label)

    ax_abs.set_xlabel(r'$\sigma$')
    ax_abs.set_ylabel(r'$\lambda_{\hat{e}}$')
    ax_abs.set_title(r'Rayleigh quotient: empirical (—) vs Wiener (- -)')
    ax_abs.legend(fontsize=8)

    ax_ratio.axhline(1.0, color='gray', lw=1, ls='--')
    ax_ratio.set_xlabel(r'$\sigma$')
    ax_ratio.set_ylabel(r'$\lambda_{\hat{e}}^{\mathrm{emp}} / \lambda_{\hat{e}}^{\mathrm{Wiener}}$')
    ax_ratio.set_title('Ratio < 1 → near-collapse; shrinking with training confirms it')
    ax_ratio.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved → {output_path}")

    # Summary table
    print("\n{:<20s}  {:>8s}  {:>14s}  {:>14s}  {:>8s}".format(
        "checkpoint", "sigma", "lambda_emp", "lambda_wiener", "ratio"))
    for label, sigmas, l_emp, l_wiener in all_results:
        for t in range(len(sigmas)):
            ratio = l_emp[t] / l_wiener[t] if abs(l_wiener[t]) > 1e-30 else float('nan')
            print("{:<20s}  {:>8.4f}  {:>14.4e}  {:>14.4e}  {:>8.4f}".format(
                label, sigmas[t], l_emp[t], l_wiener[t], ratio))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--checkpoint_names', nargs='+', default=['best_model.pth'])
    parser.add_argument('--output', default='results/jacobian_vs_wiener.pdf')
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of batches (JVP is expensive; 10 is usually enough)')
    parser.add_argument('--T', type=int, default=10,
                        help='Number of noise levels')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    data_params  = cfg['data_params']
    model_params = cfg['model_params']

    data_cfg = DataConfig(**{k: v for k, v in data_params.items() if k in _DATA_CONFIG_FIELDS})
    _, val_loader, _ = get_data_loaders(data_cfg)

    all_results = []
    for ckpt_name in args.checkpoint_names:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        print(f"\nLoading {ckpt_path} ...")
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
        ).to(device)
        model.eval()
        model.compute_schedule_variables(model.sqrtOneMinusAlphasCumprod.ravel()[-20:])

        label = os.path.splitext(ckpt_name)[0]
        sigmas, l_emp, l_wiener = collect(
            model, val_loader, device, n_batches=args.n_batches, T=args.T)
        all_results.append((label, sigmas, l_emp, l_wiener))
        del model

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plot(all_results, args.output)


if __name__ == '__main__':
    main()
