#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Project Imports ---
sys.path.append(os.getcwd()) 
sys.path.append('/mnt/SSD2/constantin/diffusion-multisteps')

from src.data_loader import get_data_loaders

def center_crop(data, target_size=(64, 64)):
    """Crops spatial dims to target_size."""
    if data.ndim < 2: return data 
    h, w = data.shape[-2], data.shape[-1]
    th, tw = target_size
    if h < th or w < tw: return data 
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return data[..., start_h:start_h+th, start_w:start_w+tw]

def compute_spectrum_1d(field):
    """Returns k, Energy for 1D field (C, L) or (L,)"""
    if field.ndim == 2: field = field[0]
    n = len(field)
    fft_vals = np.fft.rfft(field)
    energy = np.abs(fft_vals)**2
    k = np.fft.rfftfreq(n) * n
    return k, energy

def compute_spectrum_2d_radial(field, max_radius=33):
    """Returns k, Radial Energy for 2D field (C, H, W)"""
    if field.ndim == 3:
        # Sum energy across channels (e.g., u^2 + v^2)
        total_energy = 0
        for c in range(field.shape[0]):
            _, e = compute_spectrum_2d_radial(field[c], max_radius)
            if isinstance(total_energy, int): total_energy = e
            else: total_energy += e
        return np.arange(len(total_energy)), total_energy

    h, w = field.shape
    f = np.fft.fft2(field)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)**2
    
    y, x = np.indices((h, w))
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    tbin = np.bincount(r.ravel(), weights=mag.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / (nr + 1e-10)
    
    if max_radius is not None:
        radial_prof = radial_prof[:max_radius]
    
    return np.arange(len(radial_prof)), radial_prof

def main():
    parser = argparse.ArgumentParser(description="Signal vs Noise Energy Plot (Combined)")
    parser.add_argument('--configs', nargs=3, required=True, help="Paths to 3 config files")
    parser.add_argument('--log_sigmas', nargs=3, type=float, required=True, help="Sigma_t for each dataset")
    parser.add_argument('--output', default="signal_vs_noise_combined.png")
    parser.add_argument('--batches', type=int, default=20, help="Batches to average")
    
    args = parser.parse_args()
    
    # Setup single plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors for the 3 datasets
    colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
    
    print(f"--- Processing {len(args.configs)} Datasets ---")

    for i, (config_path, log_sigma) in enumerate(zip(args.configs, args.log_sigmas)):
        if not os.path.exists(config_path):
            print(f"Warning: Config {config_path} not found.")
            continue

        sigma = 10**log_sigma

        with open(config_path, 'r') as f: conf = json.load(f)
        dataset_name = conf['data_params'].get('dataset_name', f"Dataset {i+1}")
        
        print(f"[{dataset_name}] Loading... Sigma={sigma}")
        
        # Determine Alpha Bar
        # sigma = sqrt(1 - alpha_bar) => sigma^2 = 1 - alpha_bar => alpha_bar = 1 - sigma^2
        alpha_bar = 1 - sigma**2
        
        # Setup Loader
        conf['data_params']['batch_size'] = 32
        conf['data_params']['val_batch_size'] = 32
        
        try:
            _, val_loader, _ = get_data_loaders(conf['data_params'])
            iterator = iter(val_loader)
            
            accum_clean = None
            accum_noise = None
            count = 0
            
            for _ in range(args.batches):
                try: batch = next(iterator)
                except StopIteration: break
                
                # Get Clean Data
                clean_batch = batch['data'].cpu().numpy()[:, 0] # (B, C, H, W) or (B, C, L)
                
                # Generate Gaussian Noise matching shape
                noise_batch = np.random.randn(*clean_batch.shape)
                
                for b in range(clean_batch.shape[0]):
                    sample_clean = clean_batch[b]
                    sample_noise = noise_batch[b]
                    
                    # Detect dimensionality
                    if len(sample_clean.shape) == 2 and sample_clean.shape[1] > 1:
                        # 1D
                        k, E_c = compute_spectrum_1d(sample_clean)
                        _, E_n = compute_spectrum_1d(sample_noise)
                    elif len(sample_clean.shape) == 3:
                        # 2D -> Crop -> Radial
                        sample_clean = center_crop(sample_clean)
                        sample_noise = center_crop(sample_noise)
                        k, E_c = compute_spectrum_2d_radial(sample_clean, max_radius=33)
                        _, E_n = compute_spectrum_2d_radial(sample_noise, max_radius=33)
                    else:
                        continue
                        
                    if accum_clean is None:
                        accum_clean = np.zeros_like(E_c)
                        accum_noise = np.zeros_like(E_n)
                        
                    if E_c.shape == accum_clean.shape:
                        accum_clean += E_c
                        accum_noise += E_n
                        count += 1
            
            if count > 0:
                # Average
                E_clean_avg = accum_clean / count
                E_noise_avg = accum_noise / count
                
                # --- SCALING ---
                # Signal Energy scales with alpha_bar (amplitude scales with sqrt(alpha_bar))
                # Noise Energy scales with sigma^2
                # Since E ~ Amplitude^2:
                final_signal = E_clean_avg * alpha_bar
                final_noise = E_noise_avg * (sigma**2)
                
                # --- NORMALIZATION ---
                # Normalize so the CLEAN signal peak is 1. 
                # Keep noise relative to that scaling.
                peak = np.max(E_clean_avg) 
                final_signal /= peak
                final_noise /= peak
                
                # Plot Signal (Solid)
                ax.loglog(k, final_signal, label=f"{dataset_name} (Signal)", 
                          color=colors[i], linestyle='-', linewidth=2.5)
                
                # Plot Noise (Dotted)
                ax.loglog(k, final_noise, label=f"{dataset_name} (Noise $\sigma={sigma}$)", 
                          color=colors[i], linestyle=':', linewidth=2.5)
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    # Formatting
    ax.set_title("Signal vs Noise Spectral Energy", fontsize=16)
    ax.set_xlabel(r"Wavenumber $k$", fontsize=14)
    ax.set_ylabel("Normalized Energy", fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.set_ylim(bottom=1e-8)
    
    # Legend
    ax.legend(fontsize=10, loc='best', ncol=1)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()