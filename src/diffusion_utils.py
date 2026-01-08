import torch

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

def uniform_noise_beta_schedule(timesteps):
    return get_betas(torch.linspace(0.2, 1, timesteps))

def sigmoid_beta_schedule(timesteps):
    if timesteps < 20:
        raise Warning("Warning: Less than 20 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (1000/timesteps) # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000/timesteps)
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return torch.clip(betas, 0.0001, 0.9999)

def psd_beta_schedule(timesteps):
    return torch.Tensor([ 4.53887314e-4, 2.85259073e-5, 2.85267210e-5, 2.85275348e-5, 3.43625704e-5, 4.60325957e-5, 4.60347148e-5, 4.60368341e-5, 4.10608473e-5, 3.85733780e-5, 3.85748659e-5, 3.85763540e-5, 9.50367681e-5, 9.50458010e-5, 9.50548356e-5, 9.24530131e-5, 8.72393612e-5, 8.72469726e-5, 8.72545853e-5, 1.11271477e-4, 1.23289835e-4, 1.23305037e-4, 1.23320243e-4, 1.38707811e-4, 1.38727053e-4, 1.38746301e-4, 1.77679095e-4, 2.55551583e-4, 2.55616906e-4, 2.55682263e-4, 2.67478209e-4, 2.73416620e-4, 2.73491397e-4, 2.73566215e-4, 4.05702254e-4, 4.05866915e-4, 4.06031710e-4, 4.71263714e-4, 6.01681417e-4, 6.02043655e-4, 6.02406330e-4, 6.51330849e-4, 6.76051886e-4, 6.76509241e-4, 6.76967216e-4, 7.92570006e-4, 7.93198671e-4, 7.93828335e-4, 9.96352187e-4, 1.40153499e-3, 1.40350204e-3, 1.40547463e-3, 2.23737102e-3, 2.65827770e-3, 2.66536298e-3, 2.67248612e-3, 3.71762129e-3, 3.73149357e-3, 3.74546977e-3, 4.59707837e-3, 6.30109965e-3, 6.34105527e-3, 6.38152085e-3, 8.53294772e-3, 9.67068796e-3, 9.76512342e-3, 9.86142141e-3, 1.84721510e-2, 1.88197930e-2, 1.91807712e-2, 2.46602855e-2, 3.57507445e-2, 3.70762479e-2, 3.85038253e-2, 4.00154242e-2, 4.16676139e-2, 4.34792922e-2, 4.54556727e-2, 7.70881607e-2, 8.35271121e-2, 9.11397525e-2, 1.05886062e-1, 1.30967473e-1, 1.50704915e-1, 1.77447059e-1, 1.88729694e-1, 2.15995741e-1, 2.75503275e-1, 3.80268488e-1, 7.68084062e-2, 8.31987712e-2, 9.07489743e-2, 1.24347840e-1, 1.98059165e-1, 2.46974784e-1, 3.27976779e-1, 2.19751730e-1, 1.09715958e-1, 1.23237027e-1, 1.40559114e-1 ])

def log_uniform_beta_schedule(timesteps, min=-5):
    return torch.logspace(min, -0.2, 100)

def piecewise_log_beta_schedule(timesteps):
    nb_low_points = int(0.7*timesteps)
    nb_high_points = timesteps - nb_low_points
    return torch.concatenate((torch.logspace(-3.5, -2.5, nb_low_points), torch.logspace(-2.5, -0.2, nb_high_points)))

def predict_start_from_noise(x_t, t, predictedNoise, diff_model):
    return (x_t - diff_model.sqrtOneMinusAlphasCumprod[t] * predictedNoise)/diff_model.sqrtAlphasCumprod[t]
        
import numpy as np

def ddim_x0_estimate(xt, t, diff_model, denoising_model, reduced_n_steps, ddim_sampling_eta):

    cond = xt[:, :2]
    xt = xt[:, -2:]

    device = xt.device
    sampling_timesteps, eta = reduced_n_steps, ddim_sampling_eta

    batch = xt.shape[0]

    if len(t) == 1:
        batch_t = torch.ones(batch, device=device, dtype=torch.long)*t
    else:
        batch_t = t
        
    batch_t = batch_t.cpu().numpy()
    seqs = []
    seqs_next = []
    for t_idx, t in enumerate(batch_t):
        seq = list(np.linspace(0, batch_t[t_idx], sampling_timesteps+2, endpoint=True, dtype=float)) # evenly spread from 0 to current t
        seq = list(map(int, seq))
        seqs.append(list(reversed(seq)))
        seq_next = [-1] + list(seq[:-1])
        seqs_next.append(list(reversed(seq_next)))
        seq = None

    # tranpose to have time as first dimension
    cur_times = torch.tensor(seqs, device=device).T
    next_times = torch.tensor(seqs_next, device=device).T

    time_pairs = list(zip(cur_times, next_times)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    cur_x = xt
    x0_pred = None

    for t, t_next in time_pairs:
        # create mask for those timesteps that are equal
        mask = (t == t_next).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        noisy_input = torch.concat((cond, cur_x), dim=1)
        eps_theta = denoising_model(noisy_input, t)[:, -2:]
        x0_pred = predict_start_from_noise(cur_x, t, eps_theta, diff_model)

        if t_next[0] < 0: # this happens when we predict x0, should never happen during training
            # assert that all next timesteps are equal
            assert torch.all(t_next == -1), 'Next timesteps should be -1, otherwise this is inconsistent.'
            cur_x = x0_pred
            continue

        alpha = 1 - diff_model.betas[t]
        alpha_next = 1 - diff_model.betas[t_next]

        if ddim_sampling_eta != 0:
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        else:
            sigma = 0
        
        c = (1 - alpha_next).sqrt()

        noise = torch.randn_like(cur_x)

        # only change cur_x where t != t_next
        cur_x_ = x0_pred * alpha_next.sqrt() + \
                c * eps_theta + \
                sigma * noise

        cur_x = mask * cur_x + (1 - mask) * cur_x_
        
    return cur_x


"""def compute_estimate(model, k, cond, gt):
    #dNoisy = torch.randn_like(gt).to('cuda')
    with torch.no_grad():

        #interm_estimates = []
        
        dNoisy = model.sqrtAlphasCumprod[k] * gt + model.sqrtOneMinusAlphasCumprod[k] * torch.randn_like(gt).to('cuda')

        for i in reversed(range(0, k+1, 1)):


            t = i * torch.ones(dNoisy.shape[0]).to('cuda').long()
            dNoiseCond = torch.concat((cond, dNoisy), dim=1)

            predictedNoiseCond = model.unet(dNoiseCond, t)

            # use model (noise predictor) to predict mean
            modelMean = model.sqrtRecipAlphas[t] * (dNoiseCond - model.betas[t] * predictedNoiseCond / model.sqrtOneMinusAlphasCumprod[t])
            dNoisy = modelMean[:, cond.shape[1]:]
            
            dNoisy = dNoisy + model.sqrtPosteriorVariance[t] * torch.randn_like(dNoisy)

            estimate = (dNoiseCond[:, cond.shape[1]:]  - model.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:])/model.sqrtAlphasCumprod[t]

            #interm_estimates.append(estimate)

        return estimate"""
    

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
    

import torch
import torch.nn.functional as F
import numpy as np

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
