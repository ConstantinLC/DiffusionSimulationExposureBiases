#!/usr/bin/env python
"""
Evaluate the linearization of the own-prediction bias (Proposition H.5, Eq. 42).

For each noise level sigma_k, compares:
  - Empirical B^{own}(sigma) = E_own / E_clean
  - Theoretical prediction from Eq. 42:
        B^{own}(sigma) ≈ 1 + alpha_sigma * E[||J_theta||_F^2] / d

Also verifies the linearization directly (option 3):
  u_theta(hat_y^k) ≈ u_theta(y^k) + J_theta(y^k) * sqrt(alpha_k) * epsilon_clean

Usage:
  python experiments/eval_linearization.py \
      --checkpoint_dir checkpoints/KolmogorovFlow/forecasting/DiffusionModel_linear_20 \
      --output results/linearization.pdf
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.func import jvp, vmap

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
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


def build_diffusion_model(model_cfg: dict) -> DiffusionModel:
    return DiffusionModel(
        dimension=model_cfg['dimension'],
        dataSize=model_cfg['dataSize'],
        condChannels=model_cfg['condChannels'],
        dataChannels=model_cfg['dataChannels'],
        diffSchedule=model_cfg['diffSchedule'],
        diffSteps=model_cfg['diffSteps'],
        inferenceSamplingMode=model_cfg['inferenceSamplingMode'],
        inferenceConditioningIntegration=model_cfg['inferenceConditioningIntegration'],
        diffCondIntegration=model_cfg['diffCondIntegration'],
        padding_mode=model_cfg.get('padding_mode', 'circular'),
        architecture=model_cfg.get('architecture', 'ours'),
    )


# ── Core: denoiser as a pure function of the noisy data input ───────────────

def make_denoiser_fn(model, cond, t):
    """
    Returns a function  f(d_noisy) -> x0_estimate
    that wraps the UNet prediction at timestep t, with conditioning fixed.

    d_noisy has shape (C_data, *spatial) — a single sample, no batch dim.
    """
    sqrt_alpha = model.sqrtAlphasCumprod[t[0]]       # scalar
    sqrt_one_minus_alpha = model.sqrtOneMinusAlphasCumprod[t[0]]  # scalar
    C_cond = cond.shape[1]

    def denoiser(d_noisy):
        # Add batch dim
        d_noisy_b = d_noisy.unsqueeze(0)
        cond_input = torch.cat((cond, d_noisy_b), dim=1)
        pred_noise = model.unet(cond_input, t)[:, C_cond:]
        x0 = (d_noisy_b - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
        return x0.squeeze(0)

    return denoiser


# ── Hutchinson estimator for ||J||_F^2 ─────────────────────────────────────

def estimate_jacobian_frob_sq(denoiser_fn, d_noisy, n_probes=10):
    """
    Estimate ||J_theta(y_noisy)||_F^2 via Hutchinson:
        ||J||_F^2 = E_v[ ||J v||^2 ]  where v ~ N(0, I)
    Each probe is one JVP (≈ one forward pass).
    """
    norms = []
    for _ in range(n_probes):
        v = torch.randn_like(d_noisy)
        _, Jv = jvp(denoiser_fn, (d_noisy,), (v,))
        norms.append(Jv.pow(2).sum().item())
    return np.mean(norms)


# ── Direct linearization check ──────────────────────────────────────────────

def linearization_check(denoiser_fn, d_noisy_clean, sqrt_alpha, epsilon_clean):
    """
    Compare:
      actual:      u_theta(hat_y) - y
      linearized:  epsilon_clean + J_theta(y_clean) * sqrt(alpha) * epsilon_clean

    Returns (actual_error, linearized_error) as tensors.
    """
    tangent = sqrt_alpha * epsilon_clean
    u_clean, J_eps = jvp(denoiser_fn, (d_noisy_clean,), (tangent,))

    # u_theta(hat_y) via actual forward on the perturbed input
    hat_y = d_noisy_clean + tangent
    u_own = denoiser_fn(hat_y)

    return u_own, u_clean + J_eps


# ── Main evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_empirical(model, val_loader, device, n_batches=20):
    """Compute empirical B^{own} and E_clean at each noise level."""
    model.eval()
    T = model.timesteps
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu()
    mse_clean_acc = np.zeros(T)
    mse_own_acc = np.zeros(T)
    count = 0

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        data = sample['data'].to(device)
        cond = data[:, 0]
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
            ec = (x0_clean[t_idx] - target).pow(2).mean(dim=spatial_dims).mean().item()
            eo = (x0_own[t_idx] - target).pow(2).mean(dim=spatial_dims).mean().item()
            mse_clean_acc[t_idx] += ec
            mse_own_acc[t_idx] += eo
        count += 1

    mse_clean = mse_clean_acc / count
    mse_own = mse_own_acc / count
    bias_empirical = mse_own / (mse_clean + 1e-30)

    return {
        'mse_clean': mse_clean,  # indexed 0=highest sigma (reversed iteration)
        'mse_own': mse_own,
        'bias_empirical': bias_empirical,
        'sigmas': sigmas.numpy(),
    }


def evaluate_jacobian(model, val_loader, device, n_batches=5, n_probes=10):
    """
    For each noise level, compute the full decomposition of B^{own}:

        eps_own = eps_clean + Jd + R      where d = sqrt(alpha) * eps_clean

        B^{own} = ||eps_own||^2 / ||eps_clean||^2
               = 1 + (||Jd||^2 + ||R||^2
                      + 2<eps,Jd> + 2<eps,R> + 2<Jd,R>) / ||eps||^2

    We process one sample at a time (JVP is per-sample).
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    T = model.timesteps
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze()
    sqrt_alphas = model.sqrtAlphasCumprod.squeeze()

    # Accumulators for each term, all normalized by ||eps_clean||^2
    Jd_sq_acc = np.zeros(T)          # ||Jd||^2 / ||eps||^2
    R_sq_acc = np.zeros(T)           # ||R||^2 / ||eps||^2
    cross_eps_Jd_acc = np.zeros(T)   # 2<eps, Jd> / ||eps||^2
    cross_eps_R_acc = np.zeros(T)    # 2<eps, R> / ||eps||^2
    cross_Jd_R_acc = np.zeros(T)     # 2<Jd, R> / ||eps||^2
    jac_frob_sq_acc = np.zeros(T)    # Hutchinson ||J||_F^2
    eps_clean_norm_acc = np.zeros(T)  # ||eps_clean||^2 (unnormalized)
    count = 0

    dot = lambda a, b: (a * b).sum().item()

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        data = sample['data'].to(device)
        cond_batch = data[:, 0]
        target_batch = data[:, 1]

        # Process first sample in each batch
        cond = cond_batch[:1]
        target = target_batch[:1]
        y = target.squeeze(0)

        for t_idx in range(T):
            i = T - 1 - t_idx  # actual timestep (high i = high sigma)
            t = torch.full((1,), i, device=device, dtype=torch.long)

            sqrt_alpha_t = sqrt_alphas[i]
            sigma_t = sigmas[i]

            noise = torch.randn_like(y)
            y_noisy = sqrt_alpha_t * y + sigma_t * noise

            denoiser_fn = make_denoiser_fn(model, cond, t)

            # u_theta(y_noisy) -> eps_clean
            with torch.enable_grad():
                u_clean = denoiser_fn(y_noisy)
            eps_clean = u_clean - y
            eps_sq = dot(eps_clean, eps_clean)
            eps_clean_norm_acc[t_idx] += eps_sq

            # Jd = J * sqrt(alpha) * eps_clean  via JVP
            delta = sqrt_alpha_t * eps_clean
            with torch.enable_grad():
                _, Jd = jvp(denoiser_fn, (y_noisy,), (delta,))

            # u_theta(hat_y) -> actual own-pred output
            hat_y = y_noisy + delta
            with torch.enable_grad():
                u_own = denoiser_fn(hat_y)

            # Remainder R = u_own - u_clean - Jd
            R = u_own - u_clean - Jd

            # Accumulate normalized terms
            Jd_sq_acc[t_idx] += dot(Jd, Jd) / eps_sq
            R_sq_acc[t_idx] += dot(R, R) / eps_sq
            cross_eps_Jd_acc[t_idx] += 2 * dot(eps_clean, Jd) / eps_sq
            cross_eps_R_acc[t_idx] += 2 * dot(eps_clean, R) / eps_sq
            cross_Jd_R_acc[t_idx] += 2 * dot(Jd, R) / eps_sq

            # Hutchinson: ||J||_F^2
            with torch.enable_grad():
                jac_frob_sq = estimate_jacobian_frob_sq(
                    denoiser_fn, y_noisy, n_probes=n_probes
                )
            jac_frob_sq_acc[t_idx] += jac_frob_sq

        count += 1
        print(f"  Jacobian eval: batch {batch_idx+1}/{n_batches}")

    # Average
    for arr in [Jd_sq_acc, R_sq_acc, cross_eps_Jd_acc, cross_eps_R_acc,
                cross_Jd_R_acc, jac_frob_sq_acc, eps_clean_norm_acc]:
        arr /= count

    d_spatial = int(np.prod(target_batch.shape[1:]))

    alpha_sigma = 1.0 - sigmas.cpu().numpy() ** 2
    alpha_sigma_ordered = alpha_sigma[::-1].copy()

    # B^{own} predictions from different levels of the decomposition
    bias_eq42 = 1.0 + alpha_sigma_ordered * jac_frob_sq_acc / d_spatial
    bias_Jd_only = 1.0 + Jd_sq_acc
    bias_first_order = 1.0 + Jd_sq_acc + cross_eps_Jd_acc
    bias_full_linear = 1.0 + Jd_sq_acc + cross_eps_Jd_acc + R_sq_acc + cross_eps_R_acc + cross_Jd_R_acc

    return {
        # Decomposition terms (all normalized by ||eps||^2, averaged)
        'Jd_sq': Jd_sq_acc,
        'R_sq': R_sq_acc,
        'cross_eps_Jd': cross_eps_Jd_acc,
        'cross_eps_R': cross_eps_R_acc,
        'cross_Jd_R': cross_Jd_R_acc,
        # Aggregated predictions
        'bias_eq42': bias_eq42,
        'bias_Jd_only': bias_Jd_only,
        'bias_first_order': bias_first_order,
        'bias_full_linear': bias_full_linear,
        # Raw quantities
        'jac_frob_sq': jac_frob_sq_acc,
        'eps_clean_norm': eps_clean_norm_acc,
        'd_spatial': d_spatial,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(empirical, jacobian, output_path):
    T = len(empirical['sigmas'])
    sigmas_schedule = empirical['sigmas']  # schedule order: 0=lowest sigma
    sigmas_plot = sigmas_schedule[::-1]    # now 0=highest sigma, matching acc order

    bias_emp = empirical['bias_empirical']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Plot 1: B^{own} — empirical vs predictions at each level ──
    ax = axes[0]
    ax.plot(sigmas_plot, bias_emp, 'o-', color='tab:blue', linewidth=2.5,
            label=r'Empirical $\mathcal{B}^{own}$')
    ax.plot(sigmas_plot, jacobian['bias_first_order'], 's-', color='tab:green',
            label=r'$1 + \|Jd\|^2 + 2\langle\epsilon, Jd\rangle$')
    ax.plot(sigmas_plot, jacobian['bias_Jd_only'], '^--', color='tab:orange', alpha=0.7,
            label=r'$1 + \|Jd\|^2$ only')
    ax.plot(sigmas_plot, jacobian['bias_eq42'], 'v:', color='tab:red', alpha=0.5,
            label=r'Eq. 42: $1 + \alpha_\sigma \|J\|_F^2 / d$')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'Noise level $\sigma$')
    ax.set_ylabel(r'$\mathcal{B}^{own}(\sigma)$')
    ax.set_title(r'Own-prediction bias: predictions')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')

    # ── Plot 2: Decomposition — all terms stacked ──
    ax = axes[1]
    ax.plot(sigmas_plot, jacobian['cross_eps_Jd'], 'o-', color='tab:green',
            label=r'$2\langle\epsilon, Jd\rangle / \|\epsilon\|^2$')
    ax.plot(sigmas_plot, jacobian['Jd_sq'], 's-', color='tab:orange',
            label=r'$\|Jd\|^2 / \|\epsilon\|^2$')
    ax.plot(sigmas_plot, jacobian['R_sq'], '^-', color='tab:purple',
            label=r'$\|R\|^2 / \|\epsilon\|^2$')
    ax.plot(sigmas_plot, jacobian['cross_eps_R'], 'v-', color='tab:red',
            label=r'$2\langle\epsilon, R\rangle / \|\epsilon\|^2$')
    ax.plot(sigmas_plot, jacobian['cross_Jd_R'], 'D-', color='tab:gray', alpha=0.6,
            label=r'$2\langle Jd, R\rangle / \|\epsilon\|^2$')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel(r'Noise level $\sigma$')
    ax.set_ylabel(r'Contribution to $\mathcal{B}^{own} - 1$')
    ax.set_title('Decomposition of bias')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle=':')

    # ── Plot 3: Remainder cancellation ──
    ax = axes[2]
    # Show that ||R||^2 + 2<eps,R> ≈ 0 (near-perfect cancellation)
    remainder_total = jacobian['R_sq'] + jacobian['cross_eps_R'] + jacobian['cross_Jd_R']
    first_order_total = jacobian['Jd_sq'] + jacobian['cross_eps_Jd']
    ax.plot(sigmas_plot, np.abs(first_order_total), 'o-', color='tab:green',
            label=r'$|\|Jd\|^2 + 2\langle\epsilon,Jd\rangle|$ (1st order)')
    ax.plot(sigmas_plot, np.abs(remainder_total), 's-', color='tab:purple',
            label=r'$|\|R\|^2 + 2\langle\epsilon,R\rangle + 2\langle Jd,R\rangle|$ (higher)')
    ax.plot(sigmas_plot, jacobian['R_sq'], '^--', color='tab:red', alpha=0.5,
            label=r'$\|R\|^2$ (before cancellation)')
    ax.set_xlabel(r'Noise level $\sigma$')
    ax.set_ylabel('Magnitude')
    ax.set_title('1st-order vs higher-order contributions')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate linearization of own-prediction bias (Prop. H.5, Eq. 42)')
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--n_batches_empirical', type=int, default=20)
    parser.add_argument('--n_batches_jacobian', type=int, default=5)
    parser.add_argument('--n_probes', type=int, default=10,
                        help='Hutchinson probes per noise level for ||J||_F^2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/linearization.pdf')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    raw_data = config['data_params']
    data_cfg = DataConfig(**{
        k: v for k, v in raw_data.items() if k in _DATA_CONFIG_FIELDS
    })
    _, val_loader, _ = get_data_loaders(data_cfg)

    model = build_diffusion_model(config['model_params']).to(args.device)
    ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    print(f"Loaded checkpoint from {ckpt_path}")

    T = model.timesteps
    sigmas = model.sqrtOneMinusAlphasCumprod.squeeze().cpu().numpy()
    print(f"Schedule: T={T}, sigma range [{sigmas.min():.4f}, {sigmas.max():.4f}]")

    # Step 1: Empirical B^{own}
    print("\n=== Empirical own-prediction bias ===")
    empirical = evaluate_empirical(
        model, val_loader, args.device, n_batches=args.n_batches_empirical
    )

    # Step 2: Jacobian-based theoretical prediction
    print("\n=== Jacobian evaluation (Hutchinson + linearization check) ===")
    jacobian = evaluate_jacobian(
        model, val_loader, args.device,
        n_batches=args.n_batches_jacobian, n_probes=args.n_probes,
    )

    # Print summary
    sigmas_plot = sigmas[::-1]
    print(f"\n{'sigma':>10} | {'B_emp':>8} | {'B_1st':>8} | {'||Jd||2':>10} | {'<eps,Jd>':>10} | {'||R||2':>10} | {'<eps,R>':>10} | {'R_net':>10}")
    print('-' * 95)
    for i in range(T):
        r_net = jacobian['R_sq'][i] + jacobian['cross_eps_R'][i] + jacobian['cross_Jd_R'][i]
        print(f"{sigmas_plot[i]:10.4f} | "
              f"{empirical['bias_empirical'][i]:8.4f} | "
              f"{jacobian['bias_first_order'][i]:8.4f} | "
              f"{jacobian['Jd_sq'][i]:10.6f} | "
              f"{jacobian['cross_eps_Jd'][i]:10.6f} | "
              f"{jacobian['R_sq'][i]:10.6f} | "
              f"{jacobian['cross_eps_R'][i]:10.6f} | "
              f"{r_net:10.6f}")

    # Plot
    plot_results(empirical, jacobian, args.output)


if __name__ == '__main__':
    main()
