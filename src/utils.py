import torch
import numpy as np
from scipy.stats import binned_statistic
from scipy.stats import pearsonr
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_image_correlation(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError(f"Input image shapes must be identical. Got {img1.shape} and {img2.shape}")
    if img1.dim() != 4:
        raise ValueError(f"Input images must be 4-dimensional (N, C, H, W). Got {img1.dim()} dimensions.")

    if img1.device != img2.device:
        img2 = img2.to(img1.device)

    img1 = img1.float()
    img2 = img2.float()

    mean1 = img1.mean(dim=(-2, -1), keepdim=True)
    mean2 = img2.mean(dim=(-2, -1), keepdim=True)
    centered_img1 = img1 - mean1
    centered_img2 = img2 - mean2

    numerator = (centered_img1 * centered_img2).sum(dim=(-2, -1), keepdim=True)
    sum_sq_centered1 = (centered_img1**2).sum(dim=(-2, -1), keepdim=True)
    sum_sq_centered2 = (centered_img2**2).sum(dim=(-2, -1), keepdim=True)
    denominator = torch.sqrt(sum_sq_centered1 * sum_sq_centered2)

    correlation = torch.where(
        denominator == 0,
        torch.tensor(float('nan'), device=img1.device, dtype=img1.dtype),
        numerator / denominator
    )
    return torch.mean(correlation.squeeze(-1).squeeze(-1))

import torch
import numpy as np
from scipy.stats import binned_statistic

def compute_radial_psd_error(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    num_bands: int = 12,
    statistic: str = 'mean',
    device: str = 'cpu'
) -> dict:
    """
    Computes the error between predictions and ground truths in radially-averaged
    frequency bands.

    For each image, it calculates the mean power in each radial band. It then
    aggregates these results across the entire batch using the specified statistic
    (either 'mean' or 'median').
    """
    if predictions.shape != ground_truths.shape:
        raise ValueError("Predictions and ground truths must have the same shape.")

    if statistic not in ['mean', 'median']:
        raise ValueError("The 'statistic' argument for final aggregation must be either 'mean' or 'median'.")

    predictions = predictions.to(device)
    ground_truths = ground_truths.to(device)

    predictions_np = predictions.cpu().numpy()
    ground_truths_np = ground_truths.cpu().numpy()

    N, C, H, W = predictions_np.shape
    
    # Pre-calculate coordinates and radii for efficiency
    y, x = np.indices((H, W))
    center_y, center_x = H // 2, W // 2
    radii = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_radius = np.max(radii)
    bin_edges = np.linspace(0, max_radius, num_bands + 1)
    
    # Store the radially-averaged PSD for each image individually
    all_radial_psds = []
    
    # Process each image in the batch
    for n in range(N):
        for c in range(C):
            error_img = predictions_np[n, c, :, :] - ground_truths_np[n, c, :, :]
            fft_error = np.fft.fftshift(np.fft.fft2(error_img))
            psd = np.abs(fft_error)**2
            
            # For each image, ALWAYS calculate the mean PSD in the radial bins
            radial_psd_mean, _, _ = binned_statistic(
                radii.ravel(),
                psd.ravel(),
                statistic='mean',
                bins=bin_edges
            )
            all_radial_psds.append(np.nan_to_num(radial_psd_mean))

    # Convert the list of 1D arrays into a 2D array
    # The shape will be (number_of_images, num_bands)
    psd_matrix = np.array(all_radial_psds)

    # Now, aggregate across the batch of images using the chosen statistic
    if statistic == 'mean':
        final_psd_by_band = np.mean(psd_matrix, axis=0)
    elif statistic == 'median':
        final_psd_by_band = np.median(psd_matrix, axis=0)
    
    # Format the output dictionary
    band_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(num_bands)]
    result = {label: value for label, value in zip(band_labels, final_psd_by_band)}

    return result

def correlation(qa, qb):
    return pearsonr(qa.ravel(), qb.ravel())[0]

def vorticity(x: torch.Tensor) -> torch.Tensor:
    """Computes the vorticity of a 2D vector field."""
    *batch, _, h, w = x.shape
    y = x.reshape(-1, 2, h, w)
    # Pad for circular boundary conditions
    y = torch.nn.functional.pad(y, pad=(1, 1, 1, 1), mode='circular')

    du, = torch.gradient(y[:, 0], dim=-1)
    dv, = torch.gradient(y[:, 1], dim=-2)

    # Crop back to original size
    vort = (dv - du)[:, 1:-1, 1:-1]
    # Reshape to original batch dimensions
    vort = vort.reshape(*batch, h, w)
    return vort

def evaluate_trajectory_vorticity(model, loader, device, threshold=0.8):
    """
    Performs autoregressive rollout and computes vorticity correlation.
    """
    model.eval()
    all_correlations = []

    with torch.no_grad():
        for sample in loader:
            data = sample["data"].to(device)
            N, T, C, H, W = data.shape
            
            current_pred = data[:, 0] # Initial frame
            correlations_for_batch = []

            for t in range(1, T):
                # Predict the next frame
                predicted_frame = model(current_pred, None)
                ground_truth_frame = data[:, t]
                
                # Calculate vorticity for both prediction and ground truth
                pred_vort = vorticity(predicted_frame)
                gt_vort = vorticity(ground_truth_frame)
                
                # Compute correlation of the vorticities
                corr = compute_image_correlation(pred_vort.unsqueeze(1), gt_vort.unsqueeze(1))
                correlations_for_batch.append(corr.item())
                
                # The new prediction becomes the input for the next step
                current_pred = predicted_frame
            
            all_correlations.append(correlations_for_batch)

    # Average the correlations across all batches
    mean_correlations = np.mean(np.array(all_correlations), axis=0)
    
    # Find the time when correlation drops below the threshold
    drop_off_indices = np.where(mean_correlations < threshold)[0]
    time_under_threshold = drop_off_indices[0] + 1 if len(drop_off_indices) > 0 else T

    return {
        'mean_correlations': mean_correlations,
        'time_under_threshold': time_under_threshold
    }


def evaluate_trajectory_mse(model, loader, device, threshold=0.8):
    """
    Performs autoregressive rollout and computes vorticity correlation.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for sample in loader:
            data = sample["data"].to(device)
            N, T, C, H, W = data.shape
            
            current_pred = data[:, 0] # Initial frame
            errors_for_batch = []

            for t in range(1, T):
                # Predict the next frame
                predicted_frame = model(current_pred, None)
                ground_truth_frame = data[:, t]
                
                # Compute correlation of the vorticities
                error = torch.mean((predicted_frame - ground_truth_frame)**2)
                errors_for_batch.append(error.item())
                
                # The new prediction becomes the input for the next step
                current_pred = predicted_frame
            
            all_errors.append(errors_for_batch)

    # Average the correlations across all batches
    errors_per_ts = np.mean(np.array(all_errors), axis=0)
    mean_error = np.mean(errors_per_ts)

    return {
        'errors_per_ts': errors_per_ts,
        'mean_error': mean_error
    }


def separateFrequencies(data: torch.Tensor, cutoff_frequency: int = 8):
    npix = data.shape[-1]
    fft_data = torch.fft.fft2(data)
    kfreq = torch.fft.fftfreq(npix) * npix
    kfreq2D = torch.meshgrid(kfreq, kfreq)
    knrm = torch.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    fft_highpass = fft_data * (knrm > cutoff_frequency).unsqueeze(0).unsqueeze(0)
    fft_lowpass = fft_data * (knrm <= cutoff_frequency).unsqueeze(0).unsqueeze(0)
    data_highpass = torch.real(torch.fft.ifft2(fft_highpass))
    data_lowpass = torch.real(torch.fft.ifft2(fft_lowpass))
    return data_lowpass, data_highpass    


import torch

def make_freq_radius(H, W, device):
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.fftfreq(W, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    return torch.sqrt(FX**2 + FY**2)  # (H, W)

def lowpass_field(u, k_c):
    """
    Extract low-frequency component of a field.
    u: (B, C, H, W) tensor
    k_c: scalar or (B,) tensor of cutoffs
    """
    U = torch.fft.fft2(u, dim=(-2, -1))
    H, W = u.shape[-2:]
    r = make_freq_radius(H, W, u.device)  # (H, W)

    if torch.is_tensor(k_c) and k_c.ndim == 1:
        # (B, 1, H, W) mask
        mask = (r.unsqueeze(0) < k_c[:, None, None, None])  
    else:
        mask = (r < k_c).unsqueeze(0)  # broadcast across batch

    U = U * mask  # (B, C, H, W)
    return torch.fft.ifft2(U, dim=(-2, -1)).real

def highpass_field(u, k_c):
    """
    Extract high-frequency component of a field.
    u: (B, C, H, W) tensor
    k_c: scalar or (B,) tensor of cutoffs
    """
    U = torch.fft.fft2(u, dim=(-2, -1))
    H, W = u.shape[-2:]
    r = make_freq_radius(H, W, u.device)  # (H, W)

    if torch.is_tensor(k_c) and k_c.ndim == 1:
        mask = (r.unsqueeze(0) >= k_c[:, None, None, None])  
    else:
        mask = (r >= k_c).unsqueeze(0)

    U = U * mask
    return torch.fft.ifft2(U, dim=(-2, -1)).real

def shuffle_batch_dim(x):
    idx = torch.randperm(x.size(0))
    x_shuffled = x[idx]
    return x_shuffled

def get_next_run_number(base_dir):
    """
    Finds the next available run number by checking directories named 'run_*'.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('run_')]
    if not existing_runs:
        return 1
    else:
        run_numbers = [int(d.split('_')[1]) for d in existing_runs]
        return max(run_numbers) + 1
    