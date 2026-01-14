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

def low_and_high_nl_focus(timesteps, min_log_nl):
    start_log = min_log_nl
    
    n_mid = (timesteps-1)//2
    n_high = (timesteps)//2
    
    # Construct parts (handling endpoints to avoid duplicates)
    p1 = torch.tensor([10**start_log])
    p2 = torch.linspace(0.1, 0.95, n_mid + 1)[:-1]
    p3 = torch.linspace(0.95, 10**-0.0001, n_high)

    noise_levels = torch.cat((p1, p2, p3))
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


def adapt_schedule(noise_levels, weights, own_pred_errors, prev_pred_errors, clean_errors, tau, incr):
    own_ratio = own_pred_errors/clean_errors
    prev_ratio = prev_pred_errors/clean_errors

    noise_levels = noise_levels.clone()
    weights = weights.clone()
    T = len(noise_levels)
    base_weight = 0.1*T/(T-1)

    indent = 0

    for i in range(len(noise_levels)):
        if own_ratio[i] > tau:
            if i == 0:
                noise_levels[i] = noise_levels[i]*incr
            else:
                weights[i] += base_weight
                weights[0] -= base_weight
        elif prev_ratio[i] > tau:
            if i < T-1:
                new_level = (noise_levels[i+indent] + noise_levels[i+1+indent])/2
                noise_levels = torch.concatenate((noise_levels[:i+1+indent], torch.tensor([new_level]), noise_levels[i+1+indent:]))
                weights = torch.concatenate((weights[:i+1+indent], torch.tensor([base_weight]).to('cuda'), weights[i+1+indent:]))
                
                weights[0] -= base_weight
                indent += 1
    
    return noise_levels, weights

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


def find_optimal_schedule(
    model, 
    conditioning_frame, 
    target_frame,
    checkpoint_map, 
    tau=0.05, 
    device="cuda"
):
    """
    Scans for an optimized sparse schedule by finding the largest step jumps
    that maintain reconstruction error < tau.
    """
    model.eval()
    model.to(device)
    cond = conditioning_frame.to(device)
    target = target_frame.to(device)
    
    # 1. Setup Constants
    B, C, H, W = cond.shape
    sqrtOneMinusAlphas = model.sqrtOneMinusAlphasCumprod.to(device)
    sqrtAlphas = model.sqrtAlphasCumprod.to(device)
    
    # 2. Identify Valid Steps (Timesteps present in checkpoint_map)
    # We map the continuous noise levels in the checkpoint dict back to discrete integer steps
    valid_steps = []
    step_to_ckpt_map = {}

    print("Indexing available checkpoints...")
    for t in range(model.timesteps):
        current_noise_level = sqrtOneMinusAlphas[t].item()
        
        # Check if this noise level exists in the map
        found_ckpt = None
        for stored_alpha, ckpt_path in checkpoint_map.items():
            if round(stored_alpha, 5) == round(current_noise_level, 5):
                found_ckpt = ckpt_path
                break
        
        if found_ckpt:
            valid_steps.append(t)
            step_to_ckpt_map[t] = found_ckpt

    if not valid_steps:
        raise ValueError("No matching checkpoints found for model timesteps.")

    # 3. Begin Adaptive Search
    # Start from the smallest available step (cleanest)
    current_idx = 0 
    schedule = [valid_steps[0]]
    current_loaded_path = None

    print(f"Starting Adaptive Search (tau={tau}) over {len(valid_steps)} checkpoints...")

    with torch.no_grad():
        # Iterate until we reach the end of the available steps
        while current_idx < len(valid_steps) - 1:
            
            best_jump_idx = current_idx + 1
            
            # Greedily look forward to find the largest valid jump
            for test_idx in range(current_idx + 1, len(valid_steps)):
                t_val = valid_steps[test_idx]
                t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
                
                # --- A. Dynamic Checkpoint Loading ---
                target_ckpt = step_to_ckpt_map[t_val]
                
                if target_ckpt != current_loaded_path:
                    # Only load if we haven't already loaded it for this inner loop
                    state_dict = torch.load(target_ckpt, map_location=device)
                    model.unet.load_state_dict(state_dict)
                    current_loaded_path = target_ckpt

                # --- B. Simulation Step (Forward Process) ---
                # We take the CLEAN target and add noise up to level t_val
                noise = torch.randn((B, model.dataChannels, H, W), device=device)
                
                # Create the noisy input (simulating what the model would see at step t_val)
                clean_dNoise = sqrtAlphas[t_val] * target + \
                               sqrtOneMinusAlphas[t_val] * noise

                # 2. Concatenate
                clean_dNoiseCond = torch.cat((cond, clean_dNoise), dim=1)

                # 3. Predict Noise
                clean_predictedNoiseCond = model.unet(clean_dNoiseCond, t_tensor)
                
                # 4. Estimate x0 (Denoise)
                # We calculate the clean estimate to measure error
                clean_x0_estimate = (clean_dNoiseCond[:, cond.shape[1]:] - 
                                     sqrtOneMinusAlphas[t_val] * clean_predictedNoiseCond[:, cond.shape[1]:]) / sqrtAlphas[t_val]

                # --- C. Error Check ---
                clean_error = torch.mean((clean_x0_estimate - target)**2).item()

                if mse_error < tau:
                    # This jump is safe, mark it as the current best and try the next one
                    best_jump_idx = test_idx
                else:
                    # Error exceeded threshold; we cannot jump this far. 
                    # Stop looking further forward.
                    break
            
            # Commit the best jump found
            next_step = valid_steps[best_jump_idx]
            
            # Visualization log
            start_t = valid_steps[current_idx]
            print(f"Jump: {start_t} -> {next_step} | Checkpoint: {step_to_ckpt_map[next_step]}")
            
            schedule.append(next_step)
            current_idx = best_jump_idx

    print(f"Final Schedule ({len(schedule)} steps): {schedule}")
    return schedule