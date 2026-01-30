#!/usr/bin/env python
import os, argparse, json, glob, torch, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import LogFormatter
from torch import nn

sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.diffusion_utils import evaluate_dw_train_inf_gap

# --- CUSTOM SCALE FOR ERROR PLOTS ---
def get_custom_scale(axis_min=-7, axis_break=-6, axis_max=-5, break_pos=0.4):
    def forward(x):
        x = np.array(x, dtype=float); x[x <= 0] = 10**(axis_min - 1)
        lx = np.log10(x)
        norm_low = (lx - axis_min) / (axis_break - axis_min)
        norm_high = (lx - axis_break) / (axis_max - axis_break)
        return np.where(lx < axis_break, norm_low * break_pos, break_pos + norm_high * (1 - break_pos))
    def inverse(y):
        lx = np.where(y < break_pos, (y/break_pos)*(axis_break-axis_min)+axis_min, 
                      ((y-break_pos)/(1-break_pos))*(axis_max-axis_break)+axis_break)
        return 10**lx
    return forward, inverse

def format_label(name):
    if "finetuning_iteration" in name:
        num = name.replace("finetuning_iteration", "")
        return f"FT Iter {num}"
    elif "binary_search_iteration" in name:
        num = name.replace("binary_search_iteration", "")
        return f"BS Iter {num}"
    return name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--checkpoints_dir', required=True)
    parser.add_argument('--output_dir', default="./results_directory")
    parser.add_argument('--device', default="cuda")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, 'r') as f: config = json.load(f)
    accepted_names = ['binary_search_iteration0.pth', 'binary_search_iteration2.pth', 
                      'finetuning_iteration0.pth', 'finetuning_iteration1.pth']
    ckpts = sorted([f for f in glob.glob(os.path.join(args.checkpoints_dir, "*.pth")) if f.split('/')[-1] in accepted_names])
    
    if not ckpts: return print("No checkpoints found.")
    
    # 1. Load Models & Find Global Sigma Range
    models = {}
    global_min_sigma = float('inf')
    global_max_sigma = float('-inf')

    for p in ckpts:
        name = os.path.basename(p).replace('.pth', '')
        model = DiffusionModel(**{**config['model_params'], 'checkpoint': p, 'load_betas': True}).to(args.device)
        models[name] = model
        
        # Check sigmas
        alphas = model.sqrtOneMinusAlphasCumprod.ravel().cpu().numpy()
        global_min_sigma = min(global_min_sigma, np.min(alphas))
        global_max_sigma = max(global_max_sigma, np.max(alphas))

    print(f"Global Sigma Range: [{global_min_sigma:.2e}, {global_max_sigma:.2e}]")

    # 2. Evaluate
    _, val_loader, _ = get_data_loaders(config['data_params'])
    res = evaluate_dw_train_inf_gap(models, val_loader, device=args.device)

    # 3. Plotting
    n = len(models)
    
    # hspace set to 0.1 for compact, shared-axis look
    fig, axes = plt.subplots(n, 2, figsize=(7, 2.2 * n), 
                             gridspec_kw={'width_ratios': [1, 2.5], 'hspace': 0.1, 'wspace': 0.05})
    if n == 1: axes = axes[np.newaxis, :]
    
    forward, inverse = get_custom_scale(axis_min=-7, axis_break=-6, axis_max=-5, break_pos=0.4)
    
    h_clean = mlines.Line2D([], [], color='black', ls=':', lw=2, label='Clean Input')
    h_infer = mlines.Line2D([], [], color='#d62728', ls='-', lw=2, label='Inference Input')

    for i, (name, model) in enumerate(models.items()):
        alphas = model.sqrtOneMinusAlphasCumprod.ravel().cpu().numpy()
        alphas_unique = np.unique(alphas)
        sigma_min = np.min(alphas_unique)

        mse_c = res['mse_clean'][name]
        mse_a = res['mse_ancestor'][name]
        mse_o = res['mse_clean_own_pred'][name]
        alpha_list = list(alphas)[::-1] 

        ax_L, ax_R = axes[i, 0], axes[i, 1]

        # --- LEFT: Schedule Histogram ---
        bins = np.logspace(-2.5, 0, 25)
        color_hist = '#7f7f7f'
        ax_L.hist(alphas_unique, alpha=0.3, color=color_hist, bins=bins)
        ax_L.hist(alphas_unique, histtype='step', color='black', bins=bins, lw=1.2)
        ax_L.set_xscale('log')
        
        # Sigma Min INSIDE the plot (Top Center)
        ax_L.text(0.5, 0.85, f"$\sigma_{{min}} = {sigma_min:.1e}$", 
                  transform=ax_L.transAxes, ha='center', va='center', 
                  fontsize=9, fontweight='normal', 
                  bbox=dict(facecolor='white', alpha=0.6, ec='none', pad=1))
        
        short_name = format_label(name)
        ax_L.text(-0.45, 0.5, short_name, transform=ax_L.transAxes, 
                  va='center', ha='right', fontsize=10, fontweight='bold', rotation=90)
        
        if i < n-1: 
            ax_L.set_xticklabels([])

        # --- RIGHT: Error Curves ---
        ax_R.plot(alpha_list, mse_c, color='black', ls=':', lw=2)
        ax_R.plot(alpha_list, mse_a, color='#d62728', ls='-', lw=2)

        ax_R.set_yscale('function', functions=(forward, inverse))
        ax_R.yaxis.tick_right()
        ax_R.yaxis.set_label_position("right")
        
        ticks = [1e-7, 1e-6, 1e-5]
        ax_R.set_yticks(ticks)
        ax_R.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
        ax_R.set_ylim(1e-7, 1e-5)
        ax_R.grid(True, which="major", alpha=0.3, ls='--')
        ax_R.set_xscale('log')

        # *** SHARED X-AXIS LIMITS ***
        # Use global min/max so all plots align perfectly
        ax_R.set_xlim(global_min_sigma * 0.9, global_max_sigma * 1.1)

        val_inf = mse_a[-1]
        val_bias = mse_o[-1] / (mse_c[-1] + 1e-12)

        stats = [
            (r"$\mathcal{E}_{inf}(0)$", val_inf, '#d62728', ".1e"),
            (r"$\mathcal{B}^{own}(0)$", val_bias, '#1f77b4', ".2f")
        ]
        
        for j, (txt, val, c, fmt) in enumerate(stats):
            y_pos = 0.2 - (j * 0.12)
            ax_R.text(0.95, y_pos, f"{txt}={val:{fmt}}", transform=ax_R.transAxes, 
                      fontsize=12, color=c, fontweight='bold', ha='right',
                      bbox=dict(facecolor='white', alpha=0.8, ec='none', pad=0.5))

        # Hide X-ticks for all but the bottom row (Shared Axis Style)
        if i < n-1:
            ax_R.set_xticklabels([])

    # Global Legend
    fig.legend(handles=[h_clean, h_infer], 
               loc='upper center', bbox_to_anchor=(0.5, 0.93),
               fontsize=10, frameon=False, ncol=2)

    # --- CENTERED Y-LABELS ---
    big_ax = fig.add_axes([0.02, 0.05, 0.96, 0.83], frameon=False)
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    
    big_ax.set_ylabel("Schedule Density", fontsize=13, labelpad=15)
    
    right_ghost = big_ax.twinx()
    right_ghost.set_frame_on(False)
    right_ghost.set_yticks([])
    right_ghost.set_ylabel("MSE (Log Scale)", fontsize=13, rotation=270, labelpad=25)

    # --- GLOBAL X-LABELS (Bottom Row Only) ---
    axes[n-1, 0].set_xlabel(r"Noise Level $\sqrt{1-\bar{\alpha}_t}$", fontsize=9)
    axes[n-1, 1].set_xlabel(r"Noise Level $\sqrt{1-\bar{\alpha}_t}$", fontsize=9)

    plt.subplots_adjust(top=0.88, left=0.15, right=0.85)
    
    plt.savefig(os.path.join(args.output_dir, "directory_evaluation_icml_final.pdf"), bbox_inches="tight", dpi=300)
    print(f"Saved compact figure to {args.output_dir}")

if __name__ == '__main__':
    main()