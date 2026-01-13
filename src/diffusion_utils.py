import torch
import numpy as np
import torch.nn.functional as F


def get_betas(sqrtOneMinusAlphasCumprod):
        alphasCumprod = 1 - sqrtOneMinusAlphasCumprod**2
        alphas = alphasCumprod / torch.cat((torch.Tensor([1]), alphasCumprod[:-1]), dim=0)
        betas = 1 - alphas
        betas[-1] = 0.999
        return betas

### BETA SCHEDULES
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    if timesteps < 10:
        raise ValueError("Warning: Less than 10 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (500/timesteps) # adjust reference values determined for 500 steps
    beta_end = 0.02 * (500/timesteps)
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return torch.clip(betas, 0.0001, 0.9999)

def quadratic_beta_schedule(timesteps):
    if timesteps < 20:
        raise ValueError("Warning: Less than 20 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (1000/timesteps) # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000/timesteps)
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    return torch.clip(betas, 0.0001, 0.9999)

def cubic_beta_schedule(timesteps):
    if timesteps < 20:
        raise ValueError("Warning: Less than 20 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (1000/timesteps) # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000/timesteps)
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 3
    return 2*torch.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_schedule(timesteps):
    if timesteps < 20:
        raise Warning("Warning: Less than 20 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (1000/timesteps) # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000/timesteps)
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return torch.clip(betas, 0.0001, 0.9999)

def low_nl_max_out_beta_schedule(timesteps, min_log_nl):
    noise_levels = torch.linspace(10**min_log_nl, 10**(-0.0001), timesteps)
    betas = betas_from_sqrtOneMinusAlphasCumprod(noise_levels)
    return betas

def initial_exploration_beta_schedule(min_log_value, timesteps):
    start = 10**min_log_value
    end = 10**(-0.0001)

    power = 0.5
    noise_levels = torch.linspace(start**power, end**power, timesteps)**(1/power)
    betas = betas_from_sqrtOneMinusAlphasCumprod(noise_levels)
    return betas

def predict_start_from_noise(x_t, t, predictedNoise, diff_model):
    return (x_t - diff_model.sqrtOneMinusAlphasCumprod[t] * predictedNoise)/diff_model.sqrtAlphasCumprod[t]        

def compute_estimate(model, k, cond, gt):
    #dNoisy = torch.randn_like(gt).to('cuda')

    interm_estimates = []
    
    #dNoisy = torch.randn_like(gt).to('cuda')
    dNoisy = model.sqrtAlphasCumprod[k] * gt + model.sqrtOneMinusAlphasCumprod[k] * torch.randn_like(gt).to('cuda')

    for i in reversed(range(0, k+1, 1)):

        t = i * torch.ones(dNoisy.shape[0]).to('cuda').long()
        dNoiseCond = torch.concat((cond, dNoisy), dim=1)

        predictedNoiseCond = model.unet(dNoiseCond, t)

        # use model (noise predictor) to predict mean
        modelMean = model.sqrtRecipAlphas[t] * (dNoiseCond - model.betas[t] * predictedNoiseCond / model.sqrtOneMinusAlphasCumprod[t])
        dNoisy = modelMean[:, cond.shape[1]:]

        if i != 0:
        
            dNoisy = dNoisy + model.sqrtPosteriorVariance[t] * torch.randn_like(dNoisy)

        estimate = (dNoiseCond[:, cond.shape[1]:]  - model.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:])/model.sqrtAlphasCumprod[t]

        interm_estimates.append(estimate)

    return dNoisy


def betas_from_sqrtOneMinusAlphasCumprod(sqrtOneMinusAlphasCumprod: torch.Tensor) -> torch.Tensor:
    """
    Given sqrtOneMinusAlphasCumprod (shape [T] or [T, 1, 1, 1]), reconstructs a stable betas schedule.
    Ensures betas ∈ (0, 0.99].
    """
    # Flatten
    sqrtOneMinusAlphasCumprod = sqrtOneMinusAlphasCumprod.flatten().float()

    # 1. Compute alphas_cumprod
    alphas_cumprod = 1.0 - sqrtOneMinusAlphasCumprod ** 2  # shape [T]

    # Numerical safety
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1.0)

    # 2. Compute alphas from ratio
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    alphas = alphas_cumprod / alphas_cumprod_prev
    alphas = torch.clamp(alphas, min=1e-8, max=1.0)

    # 3. Compute betas
    betas = 1.0 - alphas

    # 4. Clamp betas for stability
    betas = torch.clamp(betas, min=1e-8, max=0.999)

    return betas


def run_dynamic_checkpoint_inference(
    model, 
    conditioning_frame, 
    target_frame,
    checkpoint_map, 
    ground_truth=None,
    device="cuda"
):
    """
    Runs the reverse diffusion process with dynamic checkpoint loading.
    
    Args:
        model: The initialized DiffusionModel.
        conditioning_frame: Tensor (B, C_cond, H, W).
        checkpoint_map: Dict { float_alpha : str_path }.
        ground_truth: (Optional) Tensor (B, C_data, H, W) for one-shot validation.
        device: "cuda" or "cpu".
    """
    model.eval()
    model.to(device)
    cond = conditioning_frame.to(device)
    target = target_frame.to(device)
    
    # 1. Setup Initial Noise
    B, C, H, W = cond.shape
    x0_estimate = torch.randn((B, model.dataChannels, H, W), device=device)

    current_loaded_path = None
    sqrtOneMinusAlphas = model.sqrtOneMinusAlphasCumprod.to(device)
    
    print(f"Starting Dynamic Inference over {model.timesteps} steps...")

    with torch.no_grad():
        # 2. Reverse Diffusion Loop
        for i in reversed(range(0, model.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # --- A. Dynamic Checkpoint Loading ---
            current_noise_level = sqrtOneMinusAlphas[i].item()
            
            target_ckpt = None
            for stored_alpha, ckpt_path in checkpoint_map.items():
                # MODIFIED: Compare if numbers are the same up to the 5th decimal digit
                if round(stored_alpha, 5) == round(current_noise_level, 5):
                    target_ckpt = ckpt_path
                    break
            
            if not target_ckpt:
                continue
            elif target_ckpt != current_loaded_path:
                print(f"[Step {i} | Noise {current_noise_level:.5f}] Switching weights -> {target_ckpt}")
                state_dict = torch.load(target_ckpt, map_location=device)
                model.unet.load_state_dict(state_dict)
                current_loaded_path = target_ckpt

            # --- B. Prediction Step ---
            noise = torch.randn_like(x0_estimate)

            dNoise = model.sqrtAlphasCumprod[t] * x0_estimate + \
                            model.sqrtOneMinusAlphasCumprod[t] * noise

            # 2. Concatenate
            dNoiseCond = torch.cat((cond, dNoise), dim=1)

            # 3. Predict Noise
            predictedNoiseCond = model.unet(dNoiseCond, t)
            x0_estimate = (dNoiseCond[:, cond.shape[1]:]  - model.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:])/model.sqrtAlphasCumprod[t]

            # Clean prediction
            clean_dNoise = model.sqrtAlphasCumprod[t] * target + \
                                model.sqrtOneMinusAlphasCumprod[t] * noise

            # 2. Concatenate
            clean_dNoiseCond = torch.cat((cond, clean_dNoise), dim=1)

            # 3. Predict Noise
            clean_predictedNoiseCond = model.unet(clean_dNoiseCond, t)
            clean_x0_estimate = (clean_dNoiseCond[:, cond.shape[1]:]  - model.sqrtOneMinusAlphasCumprod[t] * clean_predictedNoiseCond[:, cond.shape[1]:])/model.sqrtAlphasCumprod[t]
            
            # Clean Own Pred
            noise = torch.randn_like(x0_estimate)
            clean_dNoise = model.sqrtAlphasCumprod[t] * clean_x0_estimate + \
                                model.sqrtOneMinusAlphasCumprod[t] * noise

            # 2. Concatenate
            clean_dNoiseCond = torch.cat((cond, clean_dNoise), dim=1)

            # 3. Predict Noise
            clean_predictedNoiseCond = model.unet(clean_dNoiseCond, t)
            clean_op_x0_estimate = (clean_dNoiseCond[:, cond.shape[1]:]  - model.sqrtOneMinusAlphasCumprod[t] * clean_predictedNoiseCond[:, cond.shape[1]:])/model.sqrtAlphasCumprod[t]
            

            print(torch.mean((clean_x0_estimate-target)**2), torch.mean((clean_op_x0_estimate-target)**2)/torch.mean((clean_x0_estimate-target)**2), torch.mean((x0_estimate-target)**2))

    return clean_x0_estimate, x0_estimate