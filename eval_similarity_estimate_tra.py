import torch
import torch.nn.functional as F
import torch.fft
import matplotlib.pyplot as plt

#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch import nn

# --- Project Imports ---
from src.data_loader import get_data_loaders
from src.dataset import TurbulenceDataset
from src.data_transformations import DataParams, Transforms
from src.model_diffusion import DiffusionModel
from src.model import Unet
from src.utils import count_parameters, parse_checkpoint_args, run_model, run_model
from torch.utils.data import DataLoader, SequentialSampler

# ==========================================
# 1. FSD & Spectral Metrics Utils
# ==========================================

def _fft_magnitude(x, take_log=True):
    """Compute magnitude of 2D FFT for tensor (N,C,H,W)."""
    fft = torch.fft.fft2(x)
    mag = torch.abs(fft)
    if take_log:
        mag = torch.log1p(mag)
    return mag

def _radial_average(psd2d):
    """
    Compute radially averaged power spectrum for (N,C,H,W).
    Returns tensor of shape (N, C, Radius).
    """
    if psd2d.ndim != 4:
        raise ValueError("Expected (N,C,H,W)")
        
    N, C, H, W = psd2d.shape
    cy, cx = H // 2, W // 2
    
    y, x = torch.meshgrid(
        torch.arange(H, device=psd2d.device),
        torch.arange(W, device=psd2d.device),
        indexing='ij'
    )
    r = torch.sqrt((x - cx)**2 + (y - cy)**2).long()
    
    nbins = int(r.max()) + 1
    
    radial_profiles = []
    for n in range(N):
        ch_profiles = []
        for c in range(C):
            psd_flat = psd2d[n, c].flatten()
            r_flat = r.flatten()
            
            tbin = torch.bincount(r_flat, weights=psd_flat, minlength=nbins)
            nr = torch.bincount(r_flat, minlength=nbins).float()
            
            radial = tbin / (nr + 1e-8)
            ch_profiles.append(radial)
        radial_profiles.append(torch.stack(ch_profiles))
        
    return torch.stack(radial_profiles)

def _sqrtm_symmetric_torch(mat):
    """Compute matrix square root for a symmetric matrix."""
    L, Q = torch.linalg.eigh(mat)
    L = torch.clamp(L, min=1e-8)
    return Q @ torch.diag(torch.sqrt(L)) @ Q.mH

def _cov_torch(features, eps=1e-6):
    """Compute covariance of features (N, D)."""
    N, D = features.shape
    if N <= 1:
        return torch.eye(D, device=features.device) * eps
    
    mu = features.mean(dim=0, keepdim=True)
    centered = features - mu
    cov = (centered.T @ centered) / (N - 1)
    return cov

def frechet_distance_torch(mu1, sigma1, mu2, sigma2):
    """Squared Fréchet distance between two Multivariate Gaussians."""
    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    cov_prod = sigma1 @ sigma2
    covmean = _sqrtm_symmetric_torch(cov_prod)

    trace_term = torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)
    trace_term = torch.clamp(trace_term, min=0) 
    return diff_sq + trace_term

def calculate_fsd(real, gen, take_log=True):
    """
    Computes Fréchet Spectral Distance (FSD) based on 1D Radial Power Spectra.
    """
    real_mag = _fft_magnitude(real, take_log)**2
    gen_mag = _fft_magnitude(gen, take_log)**2
    
    feat_real = _radial_average(real_mag)
    feat_gen = _radial_average(gen_mag)
    
    N, C, R = feat_real.shape
    feat_real = feat_real.reshape(N, -1)
    feat_gen = feat_gen.reshape(N, -1)
    
    mu_r, cov_r = feat_real.mean(dim=0), _cov_torch(feat_real)
    mu_g, cov_g = feat_gen.mean(dim=0), _cov_torch(feat_gen)

    return frechet_distance_torch(mu_r, cov_r, mu_g, cov_g)

# ==========================================
# 2. Main Logic and Loop
# ==========================================

def evaluate_estimate_metrics(models, val_loader, device):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddim_max_steps = 0
    diffusion_steps = 20
    ddim_range = range(ddim_max_steps, ddim_max_steps + 10)

    metrics_storage = {
        k: {
            "grads": {"gt": [], "diffusion": [], "estimate": [], "diffusion_alt": []},
            "fsd": {"diff_est": [], "diff_diff": [], "diff_gt": []}  # Updated to store both FSDs
        }
        for k in ddim_range
    }

    # Assume 'model', 'gt', 'pred' are defined globally
    # gt shape assumption: (Batch, Sequence, Channels, H, W) or similar depending on your loader

    for t in range(diffusion_steps):
        t_tensor = torch.tensor([t], device=device).long()
        
        # --- Prepare Noise Target ---
        gt_2nd_step = torch.tensor(gt[0, :, 3], device=device) 
        dNoise = torch.randn_like(gt_2nd_step)
        
        alpha_cumprod = model.sqrtAlphasCumprod[t_tensor]
        one_minus_alpha = model.sqrtOneMinusAlphasCumprod[t_tensor]
        dNoisy = alpha_cumprod * gt_2nd_step + one_minus_alpha * dNoise

        for k in ddim_range:
            # --- 1. Prepare Inputs ---
            pred_diffusion = torch.tensor(pred[0, :, 2], device=device)
            pred_diffusion_alt = torch.tensor(pred[1, :, 2], device=device)
            
            cond_part_A = torch.tensor(gt[0, :, 0], device=device)
            cond_part_B = torch.tensor(gt[0, :, 1], device=device)
            cond_part_C = torch.tensor(gt[0, :, 2], device=device)

            cond_1st_step = torch.cat((cond_part_A, cond_part_B), dim=1)
            
            # Estimate Prediction
            with torch.no_grad():
                estimate_pred = compute_estimate(model, k, cond_1st_step, cond_part_C)
                
            # --- NEW: Compute FSDs ---
            # 1. Diffusion vs Estimate
            fsd_diff_est = calculate_fsd(pred_diffusion, estimate_pred)
            metrics_storage[k]["fsd"]["diff_est"].append(fsd_diff_est.item())

            # 2. Diffusion_1 vs Diffusion_2 (Baseline Reference)
            fsd_diff_diff = calculate_fsd(pred_diffusion, pred_diffusion_alt)
            metrics_storage[k]["fsd"]["diff_diff"].append(fsd_diff_diff.item())

            # 2. Diffusion_1 vs GT (Baseline Reference)
            fsd_diff_gt = calculate_fsd(pred_diffusion, cond_part_C)
            metrics_storage[k]["fsd"]["diff_gt"].append(fsd_diff_gt.item())
            
            # --- 2. Construct Conditioning ---
            conditions = {
                "gt": torch.cat((cond_part_B, cond_part_C), dim=1),
                "diffusion": torch.cat((cond_part_B, pred_diffusion), dim=1),
                "diffusion_alt": torch.cat((cond_part_B, pred_diffusion_alt), dim=1),
                "estimate": torch.cat((cond_part_B, estimate_pred), dim=1),
            }

            # --- 3. Forward & Backward Loop ---
            for mode_name, cond_tensor in conditions.items():
                dNoisyCond = torch.cat((cond_tensor, dNoisy), dim=1)
                dNoiseCond = torch.cat((cond_tensor, dNoise), dim=1)
                
                model.unet.zero_grad()
                predicted_noise = model.unet(dNoisyCond, t_tensor)
                
                loss = F.smooth_l1_loss(predicted_noise[:, -4:], dNoiseCond[:, -4:])
                loss.backward(retain_graph=True)
                
                grad_flat = torch.cat([
                    p.grad.detach().flatten() for p in model.unet.parameters() if p.grad is not None
                ])
                metrics_storage[k]["grads"][mode_name].append(grad_flat.cpu())


    # ==========================================
    # 3. Analysis & Plotting
    # ==========================================

    def cosine_sim(a, b):
        return F.cosine_similarity(a, b, dim=0)

    avg_cosine_per_k = []
    avg_fsd_est = []
    avg_fsd_ref = []
    avg_fsd_gt = []

    for k in ddim_range:
        # 1. Average Gradients
        avg_grads = {
            name: torch.stack(metrics_storage[k]["grads"][name], dim=0).mean(dim=0)
            for name in metrics_storage[k]["grads"]
        }
        
        # 2. Compute Similarities
        sims = [
            cosine_sim(avg_grads["gt"], avg_grads["diffusion"]),
            cosine_sim(avg_grads["gt"], avg_grads["estimate"]),
            cosine_sim(avg_grads["diffusion"], avg_grads["estimate"]),
            cosine_sim(avg_grads["diffusion"], avg_grads["diffusion_alt"])
        ]
        avg_cosine_per_k.append(sims)
        
        # 3. Average FSD
        avg_fsd_est.append(torch.tensor(metrics_storage[k]["fsd"]["diff_est"]).mean().item())
        avg_fsd_ref.append(torch.tensor(metrics_storage[k]["fsd"]["diff_diff"]).mean().item())
        avg_fsd_gt.append(torch.tensor(metrics_storage[k]["fsd"]["diff_gt"]).mean().item())

    avg_cosine_per_k = torch.tensor(avg_cosine_per_k)

    # Plotting
    plt.figure(figsize=(14, 5))

    # Plot 1: Cosine Similarity
    plt.subplot(1, 2, 1)
    labels = ['GT↔Diff', 'GT↔Est', 'Diff↔Est', 'Diff[0]↔Diff[1]']
    for i, label in enumerate(labels):
        linestyle = '--' if 'Diff[0]' in label else '-'
        plt.plot(ddim_range, avg_cosine_per_k[:, i], label=label, linestyle=linestyle)

    plt.title('Gradient Cosine Similarity')
    plt.xlabel('Step $T_P$')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: FSD
    plt.subplot(1, 2, 2)
    plt.plot(ddim_range, avg_fsd_est, label='FSD(Diff, Est)', color='purple', marker='o')
    plt.plot(ddim_range, avg_fsd_ref, label='FSD(Diff[0], Diff[1]) [Ref]', color='gray', linestyle='--', marker='x')
    plt.plot(ddim_range, avg_fsd_gt, label='FSD(Diff[0], Diff[1]) [GT]', color='blue', linestyle='--', marker='x')

    plt.title('FSD Analysis')
    plt.xlabel('Step $T_P$')
    plt.ylabel('Frechet Spectral Distance (Lower is better)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Models on Turbulence Data")
    
    # Paths
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset root")
    
    # UPDATED ARGUMENT:
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                        help="List of checkpoints. Format: 'Name=/path/to/ckpt.pth'. Space separated.")
    
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save plots and metrics")
    
    # Params
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--limit_val_trajectories', type=int, default=10) 
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 0. Parse Checkpoints into Dictionary
    checkpoints_dict = parse_checkpoint_args(args.checkpoints)
    print(f"--- Evaluating {len(checkpoints_dict)} Models ---")
    for name, path in checkpoints_dict.items():
        print(f"  > {name}: {path}")

    # 1. Load Data
    p_d_test = DataParams(batch=100, augmentations=["normalize"], sequenceLength=[(3,2)], randSeqOffset=False,
            dataSize=[128,64], dimension=2, simFields=["dens", "pres"], simParams=["mach"], normalizeMode="traMixed")
    testSet = TurbulenceDataset("Training", [args.data_path], filterTop=["128_tra"], filterSim=[[0,1,2,14,15,16,17,18]], excludefilterSim=True, filterFrame=[(0,1000)],
                        sequenceLength=p_d_test.sequenceLength, randSeqOffset=p_d_test.randSeqOffset, simFields=p_d_test.simFields, simParams=p_d_test.simParams, printLevel="sim")

    transTest = Transforms(p_d_test)
    testSet.transform = transTest
    testSampler = SequentialSampler(testSet)
    testLoader = DataLoader(testSet, sampler=testSampler, batch_size=p_d_test.batch, drop_last=True, num_workers=4)

    condChannels = 2 * (2 + len(p_d_test.simFields) + len(p_d_test.simParams))
    dataChannels = 2 + len(p_d_test.simFields) + len(p_d_test.simParams)
    print(condChannels, dataChannels)

    # 2. Load Candidate Models (Looping over Dictionary)
    models = {}
    for name, ckpt_path in checkpoints_dict.items():
        print(f"--- Loading Candidate: {name} ---")
        
        # Initialize Architecture
        model = DiffusionModel(
            dimension=2,
            dataSize=[128, 64],
            condChannels=condChannels,
            dataChannels=dataChannels,
            diffSchedule="linear",
            diffSteps=20,
            inferenceSamplingMode="ddpm",
            inferenceConditioningIntegration="clean",
            diffCondIntegration="clean",
            inferenceInitialSampling="random",
            x0_estimate_type="mean",
            architecture="ACDM"
        ).to(args.device)
        
        # Load weights
        ckpt = torch.load(ckpt_path, map_location=args.device)['stateDictDecoder']
        print(ckpt.keys())
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
            
        models[name] = model