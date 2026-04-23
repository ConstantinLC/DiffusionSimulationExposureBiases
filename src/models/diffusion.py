
### Mainly copied from https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion.py

import torch
import torch.nn as nn
from ..utils.diffusion import linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from ..utils.diffusion import cosine_beta_schedule, cubic_beta_schedule, initial_exploration_beta_schedule
from ..utils.diffusion import psd_beta_schedule, cosine_sigma_schedule, schedule_log_linear
from ..utils.diffusion import sigmas_from_betas, betas_from_sqrtOneMinusAlphasCumprod
from ..utils.diffusion import prep
from .unet_2d import Unet
from .unet_acdm import UnetACDM
from .unet_1d import Unet1D

### DIFFUSION MODEL WITH CONDITIONING
class DiffusionModel(nn.Module):

    def __init__(self, dimension, dataSize, condChannels, dataChannels, diffSchedule, diffSteps,
                 inferenceSamplingMode, inferenceConditioningIntegration, diffCondIntegration,
                 inferenceInitialSampling = "random", padding_mode='circular',
                architecture="ours", checkpoint="", load_betas=False, schedule_path=None,
                sigma_min=None, sigma_max=None):
        super(DiffusionModel, self).__init__()
        import json

        self.dimension = dimension

        self.timesteps = diffSteps
        # sampling settings
        self.inferencePosteriorSampling = "random"      # "random" or "same", ddpm only
        self.inferenceInitialSampling = inferenceInitialSampling       # "random" or "same"
        self.inferenceConditioningIntegration = inferenceConditioningIntegration # "noisy" or "clean"
        self.inferenceSamplingMode = inferenceSamplingMode # "ddpm" or "ddim"
        self.diffCondIntegration = diffCondIntegration # "noisy" or "clean"
        self.architecture = architecture
        self.use_lognormal_timesteps = (diffSchedule == "edm")
        
        if diffSchedule == "linear":
            sigmas = sigmas_from_betas(linear_beta_schedule(timesteps=self.timesteps))
        elif diffSchedule == "quadratic":
            sigmas = sigmas_from_betas(quadratic_beta_schedule(timesteps=self.timesteps))
        elif diffSchedule == "sigmoid":
            sigmas = sigmas_from_betas(sigmoid_beta_schedule(timesteps=self.timesteps))
        elif diffSchedule == "cosine":
            sigmas = sigmas_from_betas(cosine_beta_schedule(timesteps=self.timesteps))
        elif diffSchedule == "cubic":
            sigmas = sigmas_from_betas(cubic_beta_schedule(timesteps=self.timesteps))
        elif diffSchedule == "psd":
            sigmas = sigmas_from_betas(psd_beta_schedule(timesteps=self.timesteps))
            self.timesteps=100
        elif diffSchedule == "initial_exploration":
            sigmas = initial_exploration_beta_schedule(min_log_value=-2.5, timesteps=self.timesteps)
        elif diffSchedule == "log_linear":
            sigmas = schedule_log_linear(sigma_min=10**-2.5, sigma_max=10**-1.3, T=self.timesteps)
        elif diffSchedule == "edm":
            sigmas = schedule_log_linear(sigma_min=10**-2.5, sigma_max=10**-1.3, T=self.timesteps)
        elif diffSchedule == "inverseCosLog-1.875":
            sigmas = cosine_sigma_schedule(10**-1.875, 10**-0.0001, 20)
            sigmas = torch.concatenate((torch.ones(80)*sigmas[0], sigmas))
            self.timesteps=100
        elif diffSchedule == "log_uniform":
            if sigma_min is None or sigma_max is None:
                raise ValueError("diffSchedule='log_uniform' requires sigma_min and sigma_max")
            sigmas = schedule_log_linear(sigma_min=sigma_min, sigma_max=sigma_max, T=self.timesteps)
        elif "single" in diffSchedule:
            log_sigma = diffSchedule.split("_")[1]
            sigmas = torch.tensor([10 ** float(log_sigma)])
        elif diffSchedule == "from_file":
            if schedule_path is None:
                raise ValueError("diffSchedule='from_file' requires schedule_path to be set")
            with open(schedule_path) as f:
                sched_data = json.load(f)
            sigmas = torch.tensor(sched_data["schedule"], dtype=torch.float32)
            print(f"Loaded greedy schedule ({len(sigmas)} steps) from {schedule_path}")
        else:
            raise ValueError("Unknown variance schedule")
        
        self.condChannels = condChannels
        self.dataChannels = dataChannels

        if self.architecture == "ACDM":
            self.unet = UnetACDM(
                dim=dataSize[0],
                sigmas=torch.zeros(1),
                channels=condChannels+dataChannels,
                dim_mults=(1,1,1),
                use_convnext=True,
                convnext_mult=1,
            )
        elif self.dimension == 2:
            self.unet = Unet(
                dim=dataSize[0],
                sigmas=torch.zeros(1),
                channels=condChannels+dataChannels,
                dim_mults=(1,1,1),
                use_convnext=True,
                convnext_mult=1,
                padding_mode=padding_mode,
            )
        elif self.dimension == 1:
            self.unet = Unet1D(
                dim=dataSize[0],
                sigmas=torch.zeros(1),
                channels=condChannels+dataChannels,
                dim_mults=(1,1,1),
                convnext_mult=1,
                padding_mode=padding_mode,
            )
        else:
            raise ValueError(f"Unsupported dimension: {self.dimension}")
        
        if checkpoint != "":
            print('aaaa')
            print(f"Loading Checkpoint from {checkpoint}")
            ckpt = torch.load(checkpoint)
            if 'stateDictDecoder' in ckpt.keys():
                ckpt = ckpt['stateDictDecoder']
            checkpoint_unet = {key[5:]:ckpt[key] for key in ckpt if 'unet' in key and not 'sigmas' in key}
            if not 'sigmas' in checkpoint_unet.keys():
                checkpoint_unet['sigmas'] = torch.tensor([1])
            self.unet.load_state_dict(checkpoint_unet)
            if load_betas and 'betas' in ckpt:
                sigmas = sigmas_from_betas(ckpt['betas'].ravel())

        self.compute_schedule_variables(sigmas=sigmas)

    def compute_schedule_variables(self, sigmas):
        """Accept sigmas (noise levels = sqrt(1 - alphas_cumprod)) and register buffers."""
        sigmas = sigmas.ravel().float()

        sqrtOneMinusAlphasCumprod = sigmas
        sqrtAlphasCumprod = torch.sqrt(1.0 - sigmas ** 2)

        # Register buffers with correct broadcasting shape
        self.register_buffer("sqrtAlphasCumprod", prep(sqrtAlphasCumprod, self.dimension))
        self.register_buffer("sqrtOneMinusAlphasCumprod", prep(sqrtOneMinusAlphasCumprod, self.dimension))
        print(self.sqrtOneMinusAlphasCumprod.ravel())
        self.unet.sigmas = sqrtAlphasCumprod / sqrtOneMinusAlphasCumprod
        self.timesteps = len(sigmas)

    def _sample_timesteps(self, batch_size, device):
        if self.use_lognormal_timesteps:
            sigmas = self.sqrtOneMinusAlphasCumprod.ravel()
            log_sigma = torch.randn(batch_size, device=device) * 1.2 - 1.2
            sigma_samples = log_sigma.exp().clamp(sigmas.min(), sigmas.max())
            t = torch.argmin(torch.abs(sigmas.unsqueeze(0) - sigma_samples.unsqueeze(1)), dim=1)
        else:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        return t

    def forward(self, conditioning:torch.Tensor, data:torch.Tensor = None, return_x0_estimate:bool = False,
                input_type:str = "ancestor", lower_timestep_limit:int = 0, start_step:int = None) -> torch.Tensor:

        device = conditioning.device

        # Initialize data if None (Training target or Inference starting point)
        if data is None:
            if self.dimension == 1:
                N, C_cond, L = conditioning.shape
                shape_data = (N, self.dataChannels, L)
            else:
                N, C_cond, H, W = conditioning.shape
                shape_data = (N, self.dataChannels, H, W)
            data = torch.zeros(shape_data, device=device)
            
        d = data
        cond = conditioning

        # ==========================
        # TRAINING
        # ==========================
        if self.training:
            t = self._sample_timesteps(d.shape[0], device)
            dNoise = torch.randn_like(d)
            dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * dNoise
            noise = torch.cat((cond, dNoise), dim=1)
            dNoisy = torch.cat((cond, dNoisy), dim=1)

            # Predict noise
            predictedNoise = self.unet(dNoisy, t)[:, cond.shape[1]:]
            noise = noise[:, cond.shape[1]:]

            if return_x0_estimate:
                # Direct access via [t]
                return (dNoisy[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t] * predictedNoise) / self.sqrtAlphasCumprod[t]
                        
            return noise, predictedNoise

        # ==========================
        # INFERENCE
        # ==========================
        
        else:
            # Determine starting step (shortcut: start from intermediate noise level)
            effective_start = (self.timesteps - 1) if start_step is None else start_step

            # Setup initial noise at the starting noise level
            if self.inferenceInitialSampling == "random":
                eps = torch.randn_like(d)
                cNoise = torch.randn_like(cond)  # Used if needed for specialized integration
            else:
                # Expand dims for scalar broadcasting if using fixed noise
                expand_dims = [-1] * (d.ndim - 1)
                eps = torch.randn((1,) + d.shape[1:], device=device).expand(d.shape[0], *expand_dims)
                cNoise = torch.randn((1,) + cond.shape[1:], device=device).expand(cond.shape[0], *expand_dims)

            # Initialize at the correct noise level for effective_start
            sigma_start = self.sqrtOneMinusAlphasCumprod.ravel()[effective_start]
            dNoise = sigma_start * eps

            sampleStride = 1
            if return_x0_estimate:
                all_x0_estimates = []

            for i in reversed(range(0, effective_start + 1, sampleStride)):
                t = torch.full((cond.shape[0],), i, device=device, dtype=torch.long)
                condNoisy = cond

                # Optional: Pre-calculation/Manipulation of dNoise input based on input_type
                if input_type == "clean":
                    noise_temp = torch.randn_like(d)
                    dNoise = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * noise_temp

                elif input_type == "prev-pred":
                    # Look ahead to the "previous" step in diffusion terms (higher noise)
                    t_prev = torch.clamp(t + 1, max=self.timesteps - 1)
                    
                    noise_temp = torch.randn_like(d)                    
                    # Create noisy input at t_prev
                    dNoise_prev = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * noise_temp
                    if i == self.timesteps - 2:
                        dNoise_prev = torch.randn_like(dNoise_prev)
                    
                    dNoiseCond = torch.cat((condNoisy, dNoise_prev), dim=1)
                    predictedNoiseCond = self.unet(dNoiseCond, t_prev)
                    
                    # Estimate x0 from t_prev
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t_prev] * predictedNoiseCond[:, cond.shape[1]:]) / self.sqrtAlphasCumprod[t_prev]
                    
                    # Re-noise to current t
                    dNoise = self.sqrtAlphasCumprod[t] * x0_estimate + self.sqrtOneMinusAlphasCumprod[t] * torch.randn_like(d)
                
                elif input_type == "own-pred":                
                    # 1. Forward diffusion to current t
                    dNoise = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * torch.randn_like(d)
                
                    # 2. Predict x0 using current t
                    dNoiseCond = torch.cat((condNoisy, dNoise), dim=1)
                    predictedNoiseCond = self.unet(dNoiseCond, t)
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:]) / self.sqrtAlphasCumprod[t]
                    
                    # 3. Re-noise to current t (refresh noise)
                    dNoise = self.sqrtAlphasCumprod[t] * x0_estimate + self.sqrtOneMinusAlphasCumprod[t] * torch.randn_like(d)

                elif "own-pred" in input_type:    
                    n_own_preds = int(input_type.split('_')[-1])
                    # 1. Forward diffusion to current t
                    xO_estimate = d
                    for k in range(n_own_preds):
                        print(k)
                        dNoise = self.sqrtAlphasCumprod[t] * xO_estimate + self.sqrtOneMinusAlphasCumprod[t] * torch.randn_like(d)
                    
                        # 2. Predict x0 using current t
                        dNoiseCond = torch.cat((condNoisy, dNoise), dim=1)
                        predictedNoiseCond = self.unet(dNoiseCond, t)
                        x0_estimate = (dNoiseCond[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:]) / self.sqrtAlphasCumprod[t]
                        
                    # 3. Re-noise to current t (refresh noise)
                    dNoise = self.sqrtAlphasCumprod[t] * x0_estimate + self.sqrtOneMinusAlphasCumprod[t] * torch.randn_like(d)

                # --- Standard Reverse Step ---
                dNoiseCond = torch.cat((condNoisy, dNoise), dim=1)
                predictedNoiseCond = self.unet(dNoiseCond, t)

                x0_current = (dNoiseCond[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:]) / self.sqrtAlphasCumprod[t]

                if i > 0:
                    dNoise = self.sqrtAlphasCumprod[t-1] * x0_current + self.sqrtOneMinusAlphasCumprod[t-1] * torch.randn_like(x0_current)
                else:
                    dNoise = x0_current

                if return_x0_estimate:
                    # Calculate x0 estimate at current step t
                    
                    all_x0_estimates.append(x0_current)

            if return_x0_estimate:
                return dNoise, torch.flip(torch.stack(all_x0_estimates), [0])
            return dNoise


class EDMDiffusionModel(nn.Module):
    """
    EDM implementation following Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based
    Generative Models", based on https://github.com/tum-pbs/tsm-ir-diffusion.

    Training: Algorithm 1 — lognormal sigma sampling + preconditioning (c_skip, c_out, c_in, c_noise)
              + weighted MSE loss.
    Sampling: Algorithm 1 (deterministic) or Algorithm 2 (stochastic with churn), using Euler or Heun.
    """

    def __init__(self, dimension, dataSize, condChannels, dataChannels,
                 num_steps=40, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5,
                 P_mean=-1.2, P_std=1.2, rho=7.0,
                 solver='heun', stochastic=False,
                 S_churn=10.0, S_tmin=0.0, S_tmax=1e6, S_noise=1.0,
                 padding_mode='circular', architecture='Unet2D', checkpoint=''):
        super().__init__()
        self.dimension = dimension
        self.condChannels = condChannels
        self.dataChannels = dataChannels
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.rho = rho
        self.solver = solver          # 'euler' or 'heun'
        self.stochastic = stochastic  # False → deterministic (Alg 1), True → stochastic (Alg 2)
        self.S_churn = S_churn
        self.S_tmin  = S_tmin
        self.S_tmax  = S_tmax
        self.S_noise = S_noise

        if dimension == 2:
            self.unet = Unet(
                dim=dataSize[0],
                sigmas=torch.zeros(1),
                channels=condChannels + dataChannels,
                dim_mults=(1, 1, 1),
                use_convnext=True,
                convnext_mult=1,
                padding_mode=padding_mode,
            )
        elif dimension == 1:
            self.unet = Unet1D(
                dim=dataSize[0],
                sigmas=torch.zeros(1),
                channels=condChannels + dataChannels,
                dim_mults=(1, 1, 1),
                convnext_mult=1,
                padding_mode=padding_mode,
            )
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        if checkpoint:
            print(f"[EDMDiffusionModel] Loading checkpoint from {checkpoint}")
            ckpt = torch.load(checkpoint)
            if 'stateDictDecoder' in ckpt:
                ckpt = ckpt['stateDictDecoder']
            unet_state = {k[5:]: v for k, v in ckpt.items() if 'unet' in k and 'sigmas' not in k}
            if 'sigmas' not in unet_state:
                unet_state['sigmas'] = torch.tensor([1])
            self.unet.load_state_dict(unet_state)

    def _model_forward_wrapper(self, x_noisy, sigma, cond):
        """Apply EDM preconditioning and call the UNet."""
        sd2 = self.sigma_data ** 2
        s2  = sigma ** 2
        c_skip  = sd2 / (s2 + sd2)
        c_out   = sigma * self.sigma_data / (s2 + sd2).sqrt()
        c_in    = 1.0 / (sd2 + s2).sqrt()
        c_noise = sigma.log() / 4.0

        extra = [1] * (x_noisy.ndim - 1)
        c_skip_ = c_skip.view(-1, *extra)
        c_out_  = c_out.view(-1, *extra)
        c_in_   = c_in.view(-1, *extra)

        net_in  = torch.cat([cond, c_in_ * x_noisy], dim=1)
        raw_out = self.unet(net_in, c_noise)[:, self.condChannels:]

        return c_skip_ * x_noisy + c_out_ * raw_out

    def forward(self, conditioning, data=None):
        device = conditioning.device

        if data is None:
            if self.dimension == 1:
                shape_data = (conditioning.shape[0], self.dataChannels, conditioning.shape[-1])
            else:
                shape_data = (conditioning.shape[0], self.dataChannels,
                              conditioning.shape[-2], conditioning.shape[-1])
            data = torch.zeros(shape_data, device=device)

        # ==========================
        # TRAINING
        # ==========================
        if self.training:
            rnd_normal = torch.randn([data.shape[0]], device=device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

            extra = [1] * (data.ndim - 1)
            sigma_view = sigma.view(-1, *extra)
            x_noisy = data + sigma_view * torch.randn_like(data)

            D_yn = self._model_forward_wrapper(x_noisy, sigma, conditioning)
            loss = (weight.view(-1, *extra) * (D_yn - data) ** 2).mean()
            return loss

        # ==========================
        # INFERENCE
        # ==========================
        else:
            N = conditioning.shape[0]
            if self.dimension == 1:
                shape_data = (N, self.dataChannels, conditioning.shape[-1])
            else:
                shape_data = (N, self.dataChannels, conditioning.shape[-2], conditioning.shape[-1])

            # EDM rho-schedule time step discretization
            step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=device)
            t_steps = (
                self.sigma_max ** (1.0 / self.rho)
                + step_indices / (self.num_steps - 1)
                * (self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1.0 / self.rho))
            ) ** self.rho
            t_steps = torch.cat([t_steps, t_steps.new_zeros(1)])  # append t_N = 0

            x = torch.randn(shape_data, device=device).float() * float(t_steps[0])

            import math
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                t_cur  = t_cur.float()
                t_next = t_next.float()

                # Algorithm 2: stochastic churn — inject noise to increase sigma to t_hat
                if self.stochastic:
                    gamma = min(self.S_churn / self.num_steps, math.sqrt(2) - 1) \
                            if self.S_tmin <= t_cur.item() <= self.S_tmax else 0.0
                    t_hat = t_cur * (1.0 + gamma)
                    if gamma > 0:
                        noise = torch.randn_like(x) * self.S_noise
                        x = x + noise * (t_hat ** 2 - t_cur ** 2).sqrt()
                else:
                    # Algorithm 1: deterministic — no perturbation
                    t_hat = t_cur

                # First-order Euler step from t_hat → t_next
                sigma = t_hat.expand(N)
                denoised = self._model_forward_wrapper(x, sigma, conditioning)
                d_cur = (x - denoised) / t_hat
                x_next = x + (t_next - t_hat) * d_cur

                # Heun second-order correction (skip on last step where t_next = 0)
                if self.solver == 'heun' and i < self.num_steps - 1:
                    sigma_next = t_next.expand(N)
                    denoised_next = self._model_forward_wrapper(x_next, sigma_next, conditioning)
                    d_prime = (x_next - denoised_next) / t_next
                    x_next = x + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                x = x_next

            return x