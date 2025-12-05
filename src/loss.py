import torch
import torch.nn as nn
from typing import Tuple

class DifferentiableWeightedRadialFrequencyLoss(nn.Module):
    def __init__(self, band_weights, image_size: Tuple[int, int], device='cpu', ema_alpha: float = 0.9):
        super().__init__()

        H, W = image_size
        self.num_bands = len(band_weights)

        if isinstance(device, str):
            device = torch.device(device)

        if not isinstance(band_weights, torch.Tensor):
            band_weights_tensor = torch.tensor(band_weights, dtype=torch.float32, device=device)
        else:
            band_weights_tensor = band_weights.to(dtype=torch.float32, device=device)

        self.register_buffer('band_weights', band_weights_tensor)
        self.register_buffer('initial_band_weights', band_weights_tensor.clone().detach())

        self.ema_alpha = ema_alpha
        self.register_buffer('ema_avg_band_losses', torch.zeros(self.num_bands, device=device, dtype=torch.float32))
        self.ema_initialized = False

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        center_y, center_x = H // 2, W // 2

        radii = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_radius = torch.max(radii)

        bin_edges = torch.linspace(0, max_radius, self.num_bands + 1, device=device)
        bin_indices = torch.bucketize(radii, bin_edges, right=True) - 1
        bin_indices_clamped = torch.clamp(bin_indices, 0, self.num_bands - 1)

        masks = []
        for b in range(self.num_bands):
            mask_band_b = (bin_indices_clamped == b).float()
            masks.append(mask_band_b)
        self.register_buffer('band_masks', torch.stack(masks))

        self.H = H
        self.W = W

    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        N, C, H_current, W_current = predictions.shape
        if H_current != self.H or W_current != self.W:
            raise ValueError(
                f"Input image size ({H_current}, {W_current}) does NOT match initialized size ({self.H}, {self.W}). "
                "The loss function's internal buffers (bin masks) are pre-computed for the initialized size."
            )

        error_signal = predictions - ground_truths
        fft_error_complex = torch.fft.fftshift(
            torch.fft.fft2(error_signal, dim=(-2, -1), norm='ortho'),
            dim=(-2, -1)
        )

        total_weighted_loss = 0.0
        for band_idx in range(self.num_bands):
            current_band_mask = self.band_masks[band_idx].unsqueeze(0).unsqueeze(0)
            band_fft_error_masked = fft_error_complex * current_band_mask
            band_fft_error_unshifted = torch.fft.ifftshift(band_fft_error_masked, dim=(-2, -1))
            band_error_image_complex = torch.fft.ifft2(band_fft_error_unshifted, dim=(-2, -1), norm='ortho')
            squared_error_for_band = torch.abs(band_error_image_complex)**2
            mse_for_this_band = torch.mean(squared_error_for_band)
            weighted_mse_for_band = mse_for_this_band * self.band_weights[band_idx]
            total_weighted_loss += weighted_mse_for_band
        
        final_loss = total_weighted_loss
        return final_loss

    def get_unweighted_band_losses(self, predictions, ground_truths):
        running_val_band_losses_per_epoch = []
        error_signal_val = predictions - ground_truths
        fft_error_complex_val = torch.fft.fftshift(
            torch.fft.fft2(error_signal_val, dim=(-2, -1), norm='ortho'),
            dim=(-2, -1)
        )
        unweighted_band_mses_current_batch = []
        for band_idx in range(self.num_bands):
            current_band_mask_val = self.band_masks[band_idx].unsqueeze(0).unsqueeze(0)
            band_fft_error_masked_val = fft_error_complex_val * current_band_mask_val
            band_fft_error_unshifted_val = torch.fft.ifftshift(band_fft_error_masked_val, dim=(-2, -1))
            band_error_image_complex_val = torch.fft.ifft2(band_fft_error_unshifted_val, dim=(-2, -1), norm='ortho')
            squared_error_for_band_val = torch.abs(band_error_image_complex_val)**2
            mse_for_this_band_val = torch.mean(squared_error_for_band_val)
            unweighted_band_mses_current_batch.append(mse_for_this_band_val)
        
        running_val_band_losses_per_epoch.append(torch.stack(unweighted_band_mses_current_batch))
        return running_val_band_losses_per_epoch