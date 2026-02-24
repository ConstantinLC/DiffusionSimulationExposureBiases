#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fft, fft2, fftshift

import sys
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

# --- Project Imports ---
from src.data_loader import get_data_loaders
from src.model_diffusion import DiffusionModel
from src.utils import parse_checkpoint_args

# --- Matplotlib Configuration for "Academic Paper" Style ---
# Updates: Significantly increased font sizes for better visibility
plt.rcParams['font.size'] = 20          # Increased from 16 to 20
plt.rcParams['axes.linewidth'] = 2.0    # Thicker axes
plt.rcParams['axes.labelsize'] = 22     # Axis labels larger
plt.rcParams['xtick.labelsize'] = 18    # Tick labels larger
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16    # Legend significantly larger
plt.rcParams['lines.linewidth'] = 3.5   # Thicker lines
plt.rcParams['lines.markersize'] = 11   # Larger markers
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def compute_psd(images):
    """ Robust Normalized PSD computation for 1D and 2D data. """
    device = images.device
    
    # CASE 1: 1D Data (B, C, L)
    if images.ndim == 3:
        B, C, L = images.shape
        f_transform = fft(images, dim=-1)
        power = torch.abs(f_transform)**2
        power = power.sum(dim=1) 
        power_avg = power.mean(dim=0)
        
        n_bins = L // 2
        psd = power_avg[:n_bins]
        freqs = np.arange(n_bins)
        
        # Normalize (Sum = 1)
        psd = psd / (psd.sum() + 1e-10)
        return freqs, psd.cpu().numpy()

    # CASE 2: 2D Data (B, C, H, W)
    elif images.ndim == 4:
        images_sq = images.pow(2).sum(dim=1).sqrt() 
        B, H, W = images_sq.shape

        f_transform = fft2(images_sq)
        f_shift = fftshift(f_transform, dim=(-2, -1))
        magnitude = torch.abs(f_shift) ** 2
        
        y, x = np.indices((H, W))
        center = np.array([(H-1)/2, (W-1)/2])
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = torch.from_numpy(r.astype(int)).to(device).view(-1)
        
        mag_flat = magnitude.reshape(B, -1).mean(dim=0) 
        nr = min(H, W) // 2
        
        tbin = torch.zeros(nr + 1, device=device)
        nr_bin = torch.zeros(nr + 1, device=device)
        
        mask = r <= nr
        tbin.index_add_(0, r[mask], mag_flat[mask])
        nr_bin.index_add_(0, r[mask], torch.ones_like(r[mask], dtype=torch.float))
        
        radial_profile = tbin[:nr] / (nr_bin[:nr] + 1e-8)
        freqs = np.arange(nr)
        
        # Normalize (Sum = 1)
        radial_profile = radial_profile / (radial_profile.sum() + 1e-10)
        return freqs, radial_profile.cpu().numpy()
    
    else:
        raise ValueError(f"Unsupported data shape: {images.shape}")

def evaluate_models(models, loaders, device):
    results = {}
    
    for name, model in models.items():
        print(f"--- Evaluating {name} ---")
        model.eval()
        loader = loaders[name]
        
        mse_clean_batches = []
        mse_inf_batches = []
        psd_accum = 0
        psd_count = 0
        
        with torch.no_grad():
            alphas = model.sqrtOneMinusAlphasCumprod.flatten().cpu().numpy()[::-1]

            for batch_idx, sample in enumerate(loader):
                data = sample["data"].to(device)
                cond = data[:, 0]
                target = data[:, 1]
                
                # 1. Error Curves
                _, x0_clean = model(conditioning=cond, data=target, return_x0_estimate=True, input_type="clean")
                _, x0_inf = model(conditioning=cond, data=target, return_x0_estimate=True, input_type="ancestor")
                
                mse_clean = torch.stack([torch.mean((est - target)**2) for est in x0_clean])
                mse_inf = torch.stack([torch.mean((est - target)**2) for est in x0_inf])
                
                mse_clean_batches.append(mse_clean)
                mse_inf_batches.append(mse_inf)
                
                # 2. PSD
                freqs, psd_val = compute_psd(target)
                psd_accum += psd_val
                psd_count += 1
                
                if batch_idx >= 10: break 
        
        mse_clean_avg = torch.stack(mse_clean_batches).mean(dim=0).flatten().cpu().numpy()
        mse_inf_avg = torch.stack(mse_inf_batches).mean(dim=0).flatten().cpu().numpy()

        results[name] = {
            "mse_clean": mse_clean_avg,
            "mse_inf": mse_inf_avg,
            "psd": psd_accum / psd_count,
            "freqs": freqs,
            "alphas": alphas
        }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='multi_model_analysis.pdf')
    args = parser.parse_args()
    
    ckpt_map = parse_checkpoint_args(args.checkpoints)
    model_names = list(ckpt_map.keys())
    models = {}
    loaders = {}
    
    # Load Models
    for i, name in enumerate(model_names):
        print(f"Loading {name}...")
        cfg_path = args.configs[i]
        ckpt_path = ckpt_map[name]
        
        with open(cfg_path, 'r') as f: config = json.load(f)
        config['data_params']['batch_size'] = 16
        
        _, _, test_loader = get_data_loaders(config['data_params'])
        loaders[name] = test_loader
        
        m_params = config['model_params']
        m_params['checkpoint'] = ckpt_path
        m_params['load_betas'] = True
        
        model = DiffusionModel(**m_params).to(args.device)
        models[name] = model

    # Run Eval
    data = evaluate_models(models, loaders, args.device)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8)) # Increased figure size for larger fonts
    
    # Define distinct colors for the models (Tableau 10 palette)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    ax_err = axes[0]
    ax_psd = axes[1]
    
    for i, name in enumerate(model_names):
        res = data[name]
        c = colors[i % len(colors)]
        
        # --- PLOT 1: ERROR GAP ---
        ax_err.plot(res['alphas'], res['mse_clean'], 
                    color=c, linestyle=':', linewidth=3.5,
                    marker='o', markersize=11, markeredgecolor='white', markeredgewidth=2.0,
                    label=f"{name} (Clean)")
        
        ax_err.plot(res['alphas'], res['mse_inf'], 
                    color=c, linestyle='-', linewidth=3.5,
                    marker='o', markersize=11, markeredgecolor='white', markeredgewidth=2.0,
                    label=f"{name} (Inference)")
        
        # --- PLOT 2: PSD ---
        ax_psd.plot(res['freqs'], res['psd'], 
                    color=c, linestyle='-', linewidth=3.5,
                    label=f"{name}")

    # --- Formatting Ax 1 (Error) ---
    ax_err.set_xlabel(r"Noise Level $\sigma$")
    ax_err.set_ylabel("MSE")
    ax_err.set_xscale('log')
    ax_err.set_yscale('log')
    
    # Specific Dotted Grid Style
    ax_err.grid(True, which="major", color='#999999', linestyle=':', linewidth=1.5, alpha=0.6)
    ax_err.grid(True, which="minor", color='#dddddd', linestyle=':', linewidth=1.0, alpha=0.4)
    
    # Legend Box with black border
    ax_err.legend(fancybox=False, edgecolor='black', framealpha=1.0, loc='upper left')

    # --- Formatting Ax 2 (PSD) ---
    ax_psd.set_xlabel("Wavenumber $k$")
    ax_psd.set_ylabel(r"Normalised $E(k)$")
    ax_psd.set_xscale('log')
    ax_psd.set_yscale('log')
    ax_psd.set_ylim(bottom=1e-9)
    
    ax_psd.grid(True, which="major", color='#999999', linestyle=':', linewidth=1.5, alpha=0.6)
    ax_psd.grid(True, which="minor", color='#dddddd', linestyle=':', linewidth=1.0, alpha=0.4)
    
    ax_psd.legend(fancybox=False, edgecolor='black', framealpha=1.0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Analysis saved to {args.output}")

if __name__ == "__main__":
    main()