### Mainly copied from https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion_utils import linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from src.diffusion_utils import cosine_beta_schedule, cubic_beta_schedule, initial_exploration_beta_schedule
from src.diffusion_utils import low_nl_max_out_beta_schedule, low_and_high_nl_focus, psd_beta_schedule
from src.diffusion_utils import betas_from_sqrtOneMinusAlphasCumprod
from src.model import Unet
from src.model_acdm_arch import UnetACDM
from src.model_1d import Unet1D

### DIFFUSION MODEL WITH CONDITIONING
class DiffusionModel(nn.Module):

    def __init__(self, dimension, dataSize, condChannels, dataChannels, diffSchedule, diffSteps, 
                 inferenceSamplingMode, inferenceConditioningIntegration, diffCondIntegration, 
                 inferenceInitialSampling = "random", padding_mode='circular',
                architecture="ours"):
        super(DiffusionModel, self).__init__()

        self.dimension = dimension

        self.timesteps = diffSteps
        # sampling settings
        self.inferencePosteriorSampling = "random"      # "random" or "same", ddpm only
        self.inferenceInitialSampling = inferenceInitialSampling       # "random" or "same"
        self.inferenceConditioningIntegration = inferenceConditioningIntegration # "noisy" or "clean"
        self.inferenceSamplingMode = inferenceSamplingMode # "ddpm" or "ddim"
        self.diffCondIntegration = diffCondIntegration # "noisy" or "clean"
        self.architecture = architecture
        
        self.weights = torch.ones(self.timesteps).to('cuda')

        if diffSchedule == "linear":
            betas = linear_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "cosine":
            betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "cubic":
            betas = cubic_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "psd":
            betas = psd_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "transonicIteration7":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.cat((torch.logspace(-1.2, -1, 14), torch.logspace(-0.16, -0.00005, 6))))
        elif diffSchedule == "Low-Noise-Levels-Focus":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.tensor([0.0316, 0.0350, 0.0380, 0.0407, 0.0432, 0.0455, 0.0477, 0.0498, 0.0519,
                                                                        0.0538, 0.0558, 0.0576, 0.0595, 0.0614, 0.0632, 0.0650, 0.0669, 0.0688,
                                                                        0.0706, 0.0725, 0.0745, 0.0764, 0.0785, 0.0805, 0.0826, 0.0848, 0.0870,
                                                                        0.0892, 0.0916, 0.0940, 0.0965, 0.0991, 0.1018, 0.1045, 0.1074, 0.1104,
                                                                        0.1135, 0.1168, 0.1201, 0.1237, 0.1273, 0.1312, 0.1352, 0.1393, 0.1437,
                                                                        0.1483, 0.1530, 0.1580, 0.1632, 0.1687, 0.1744, 0.1804, 0.1866, 0.1931,
                                                                        0.2000, 0.2071, 0.2146, 0.2224, 0.2305, 0.2390, 0.2479, 0.2572, 0.2669,
                                                                        0.2770, 0.2875, 0.2984, 0.3098, 0.3217, 0.3340, 0.3468, 0.3601, 0.3740,
                                                                        0.3883, 0.4032, 0.4186, 0.4345, 0.4510, 0.4681, 0.4857, 0.5040, 0.5228,
                                                                        0.5422, 0.5622, 0.5829, 0.6041, 0.6260, 0.6485, 0.6716, 0.6954, 0.7198,
                                                                        0.7449, 0.7706, 0.7969, 0.8240, 0.8516, 0.8800, 0.9090, 0.9387, 0.9690,
                                                                        0.9997], dtype=torch.float64))

        elif diffSchedule == "Mid-Noise-Levels-Focus":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.tensor([0.0316, 0.0490, 0.0609, 0.0702, 0.0781, 0.0849, 0.0910, 0.0966, 0.1018,
                                                            0.1067, 0.1113, 0.1156, 0.1198, 0.1239, 0.1278, 0.1317, 0.1355, 0.1392,
                                                            0.1428, 0.1464, 0.1500, 0.1536, 0.1571, 0.1607, 0.1642, 0.1678, 0.1714,
                                                            0.1750, 0.1786, 0.1823, 0.1860, 0.1898, 0.1936, 0.1975, 0.2014, 0.2054,
                                                            0.2095, 0.2137, 0.2179, 0.2223, 0.2267, 0.2313, 0.2360, 0.2408, 0.2457,
                                                            0.2507, 0.2559, 0.2612, 0.2667, 0.2724, 0.2782, 0.2842, 0.2904, 0.2967,
                                                            0.3033, 0.3101, 0.3171, 0.3243, 0.3318, 0.3396, 0.3476, 0.3558, 0.3644,
                                                            0.3733, 0.3824, 0.3919, 0.4017, 0.4119, 0.4224, 0.4333, 0.4445, 0.4562,
                                                            0.4683, 0.4808, 0.4937, 0.5071, 0.5209, 0.5352, 0.5500, 0.5654, 0.5812,
                                                            0.5976, 0.6145, 0.6320, 0.6501, 0.6687, 0.6880, 0.7079, 0.7284, 0.7496,
                                                            0.7714, 0.7939, 0.8172, 0.8411, 0.8657, 0.8910, 0.9171, 0.9440, 0.9716,
                                                            0.9997], dtype=torch.float64))
        
        elif diffSchedule == "High-Noise-Levels-Focus":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.tensor([0.0316, 0.0837, 0.1139, 0.1365, 0.1551, 0.1710, 0.1851, 0.1977, 0.2093,
                                                                0.2201, 0.2301, 0.2395, 0.2485, 0.2570, 0.2652, 0.2731, 0.2807, 0.2881,
                                                                0.2953, 0.3023, 0.3091, 0.3158, 0.3224, 0.3288, 0.3352, 0.3415, 0.3477,
                                                                0.3539, 0.3600, 0.3661, 0.3721, 0.3781, 0.3841, 0.3901, 0.3960, 0.4020,
                                                                0.4080, 0.4140, 0.4199, 0.4260, 0.4320, 0.4380, 0.4441, 0.4503, 0.4565,
                                                                0.4627, 0.4689, 0.4753, 0.4817, 0.4881, 0.4946, 0.5012, 0.5079, 0.5146,
                                                                0.5215, 0.5284, 0.5354, 0.5425, 0.5497, 0.5570, 0.5645, 0.5720, 0.5797,
                                                                0.5875, 0.5954, 0.6034, 0.6116, 0.6200, 0.6285, 0.6371, 0.6459, 0.6549,
                                                                0.6640, 0.6734, 0.6829, 0.6926, 0.7025, 0.7126, 0.7229, 0.7334, 0.7441,
                                                                0.7551, 0.7663, 0.7778, 0.7894, 0.8014, 0.8136, 0.8261, 0.8389, 0.8519,
                                                                0.8653, 0.8789, 0.8929, 0.9071, 0.9218, 0.9367, 0.9520, 0.9676, 0.9836,
                                                                0.9997], dtype=torch.float64))
        elif diffSchedule == "Log-2-Min":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.8, 85),
                            torch.logspace(-1.8, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
            self.timesteps=100
        elif diffSchedule == "Log-2-Min+":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.99, 85),
                            torch.logspace(-1.99, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
            self.timesteps=100
        elif diffSchedule == "Log-2-Min+Closeto1":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.99, 85),
                            torch.logspace(-1.99, -0.1, 10),
                            torch.logspace(-0.1, -0.000001, 5))))
            self.timesteps=100
        elif diffSchedule == "Log-1.5-Min+":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.5, -1.49, 85),
                            torch.logspace(-1.49, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
            self.timesteps=100
        elif diffSchedule == "Log-1-Min+":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1, -0.99, 85),
                            torch.logspace(-0.99, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
            self.timesteps=100
        elif diffSchedule == "initial_exploration":
            betas = initial_exploration_beta_schedule(min_log_value=-2.2, timesteps=self.timesteps)
        elif "lowNlMaxOut" in diffSchedule:
            min_log_nl = float(diffSchedule.split("_")[1])
            betas = low_nl_max_out_beta_schedule(timesteps=self.timesteps, min_log_nl=min_log_nl)
            self.weights = torch.concatenate((torch.tensor([0.9*self.timesteps]), 0.1*self.timesteps/(self.timesteps-1)*torch.ones(self.timesteps-1))) # Sum of weights should be equal to self.timesteps
            self.weights = self.weights.to('cuda')
        elif "lowAndHighFocus" in diffSchedule:
            min_log_nl = float(diffSchedule.split("_")[1])
            betas = low_and_high_nl_focus(timesteps=self.timesteps, min_log_nl=min_log_nl)
            self.weights = torch.concatenate((torch.tensor([0.9*self.timesteps]), 0.1*self.timesteps/(self.timesteps-1)*torch.ones(self.timesteps-1))) # Sum of weights should be equal to self.timesteps
            self.weights = self.weights.to('cuda')
        else:
            raise ValueError("Unknown variance schedule")

        self.condChannels = condChannels
        self.dataChannels = dataChannels
        if self.architecture == "ACDM":
            self.unet = UnetACDM(dim=128,
                channels= condChannels+dataChannels,
                sigmas=torch.zeros(1),
                dim_mults=(1,1,1),
                use_convnext=True,
                convnext_mult=1)
        
        elif self.architecture == "Unet2D":
            self.unet = Unet(
            dim=dataSize[0],
            sigmas=torch.zeros(1),
            channels=condChannels+dataChannels,
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
            padding_mode=padding_mode
            )

        elif self.architecture == "Unet1D":
            self.unet = Unet1D(
            dim=dataSize[0],
            sigmas=torch.zeros(1),
            channels=condChannels+dataChannels,
            dim_mults=(1,1,1),
            convnext_mult=1,
            padding_mode=padding_mode
            )
        
        else : raise Exception

        self.compute_schedule_variables(betas)

    def compute_schedule_variables(self, betas=None, sigmas = None):
        if betas is None and sigmas is None:
            raise Exception
        elif sigmas is not None:
            betas = betas_from_sqrtOneMinusAlphasCumprod(sigmas)
        betas = betas.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        alphas = 1.0 - betas
        alphasCumprod = torch.cumprod(alphas, axis=0)
        alphasCumprodPrev = F.pad(alphasCumprod[:-1], (0,0,0,0,0,0,1,0), value=1.0)
        sqrtRecipAlphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrtAlphasCumprod = torch.sqrt(alphasCumprod)
        sqrtOneMinusAlphasCumprod = torch.sqrt(1. - alphasCumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posteriorVariance = betas * (1. - alphasCumprodPrev) / (1. - alphasCumprod)
        sqrtPosteriorVariance = torch.sqrt(posteriorVariance)

        self.betas = betas.to('cuda')
        self.sqrtRecipAlphas = sqrtRecipAlphas.to('cuda')
        self.sqrtAlphasCumprod = sqrtAlphasCumprod.to('cuda')
        self.sqrtOneMinusAlphasCumprod = sqrtOneMinusAlphasCumprod.to('cuda')
        self.sqrtPosteriorVariance = sqrtPosteriorVariance.to('cuda')

        self.unet.sigmas = (self.sqrtAlphasCumprod/self.sqrtOneMinusAlphasCumprod).ravel()
        self.timesteps = len(self.betas.ravel())

    def delete_steps(self, steps_to_delete):
        print(self.sqrtOneMinusAlphasCumprod.ravel())
        retain_indices = [idx for idx in range(self.timesteps) if idx not in steps_to_delete]
        new_noise_levels = self.sqrtOneMinusAlphasCumprod.ravel()[retain_indices]
        print(new_noise_levels)
        new_betas = betas_from_sqrtOneMinusAlphasCumprod(new_noise_levels)
        self.compute_schedule_variables(new_betas)

    def forward(self, conditioning:torch.Tensor, data:torch.Tensor = None, return_x0_estimate:bool = False, 
                input_type:str = "ancestor", 
                fixed_timestep:int = None) -> torch.Tensor:

        device = "cuda" if conditioning.is_cuda else "cpu"
        
        # Determine dimensions based on input rank (3D tensor = 1D spatial, 4D tensor = 2D spatial)
        # Assuming input is (Batch, Channel, Spatial...)
        is_1d = (conditioning.ndim == 3) or (self.dimension == 1)
        
        # Define broadcasting shape for scalar multiplication
        # 1D: (B, C, L) -> scalars need shape (B, 1, 1)
        # 2D: (B, C, H, W) -> scalars need shape (B, 1, 1, 1)
        view_shape = (-1, 1, 1) if is_1d else (-1, 1, 1, 1)

        if data is None:
            if is_1d:
                N, C_cond, L = conditioning.shape
                data = torch.zeros((N, self.dataChannels, L)).to(device)
            else:
                N, C_cond, H, W = conditioning.shape
                data = torch.zeros((N, self.dataChannels, H, W)).to(device)

        if self.dimension == 3:
            raise NotImplementedError()
            
        # combine batch and sequence dimension for decoder processing
        d = data
        cond = conditioning

        # Helper to reshape schedule params for broadcasting
        def extract(a, t):
            return a[t].view(view_shape)

        # TRAINING
        if self.training:

            if fixed_timestep is None:
                lower_limit_timesteps = 0
                upper_limit_timesteps = self.timesteps

            else:
                lower_limit_timesteps = fixed_timestep    
                upper_limit_timesteps = fixed_timestep + 1

            t = torch.randint(lower_limit_timesteps, upper_limit_timesteps, (d.shape[0],), device=device).long()
            
            # Pre-fetch schedule values with correct shape
            sqrt_alphas = extract(self.sqrtAlphasCumprod, t)
            sqrt_one_minus_alphas = extract(self.sqrtOneMinusAlphasCumprod, t)

            if "input-own-pred" in input_type:
                # forward diffusion process that adds noise to data
                if self.diffCondIntegration == "noisy":
                    d = torch.concat((cond, d), dim=1)
                    noise = torch.randn_like(d, device=device)
                    dNoisy = sqrt_alphas * d + sqrt_one_minus_alphas * noise

                elif self.diffCondIntegration == "clean":
                    dNoise = torch.randn_like(d, device=device)
                    dNoisy = sqrt_alphas * d + sqrt_one_minus_alphas * dNoise

                    noise = torch.concat((cond, dNoise), dim=1)
                    dNoisy = torch.concat((cond, dNoisy), dim=1)
            
                predictedNoiseCond = self.unet(dNoisy, t)
                predictedNoise = predictedNoiseCond[:, cond.shape[1]:]

                first_estimate = (dNoisy[:, cond.shape[1]:]  - sqrt_one_minus_alphas * predictedNoise)/sqrt_alphas
                d = first_estimate
            
            elif "input-prev-pred" in input_type:
                # forward diffusion process that adds noise to data
                t_prev = t+1
                sqrt_alphas_prev = extract(self.sqrtAlphasCumprod, t_prev)
                sqrt_one_minus_alphas_prev = extract(self.sqrtOneMinusAlphasCumprod, t_prev)

                if self.diffCondIntegration == "noisy":
                    d = torch.concat((cond, d), dim=1)
                    noise = torch.randn_like(d, device=device)
                    dNoisy = sqrt_alphas_prev * d + sqrt_one_minus_alphas_prev * noise

                elif self.diffCondIntegration == "clean":
                    dNoise = torch.randn_like(d, device=device)
                    dNoisy = sqrt_alphas_prev * d + sqrt_one_minus_alphas_prev * dNoise

                    noise = torch.concat((cond, dNoise), dim=1)
                    dNoisy = torch.concat((cond, dNoisy), dim=1)
            
                predictedNoiseCond = self.unet(dNoisy, t_prev)
                predictedNoise = predictedNoiseCond[:, cond.shape[1]:]

                first_estimate = (dNoisy[:, cond.shape[1]:]  - sqrt_one_minus_alphas_prev * predictedNoise)/sqrt_alphas_prev
                d = first_estimate

            # forward diffusion process that adds noise to data
            if self.diffCondIntegration == "noisy":
                d = torch.concat((cond, d), dim=1)
                noise = torch.randn_like(d, device=device)
                dNoisy = sqrt_alphas * d + sqrt_one_minus_alphas * noise

            elif self.diffCondIntegration == "clean":
                dNoise = torch.randn_like(d, device=device)
                dNoisy = sqrt_alphas * d + sqrt_one_minus_alphas * dNoise

                noise = torch.concat((cond, dNoise), dim=1)
                dNoisy = torch.concat((cond, dNoisy), dim=1)

            else:
                raise ValueError("Unknown conditioning integration mode")

            # noise prediction with network
            predictedNoise = self.unet(dNoisy, t)[:, cond.shape[1]:]
            noise = noise[:, cond.shape[1]:]
            # unstack batch and sequence dimension again

            weights = self.weights[t]
            # Use dynamic view shape here
            scale_factor = torch.sqrt(weights).view(view_shape)

            if return_x0_estimate:
                x0_estimate = (dNoisy[:, cond.shape[1]:]  - sqrt_one_minus_alphas * predictedNoise)/sqrt_alphas
                return x0_estimate
                        
            return scale_factor*noise, scale_factor*predictedNoise


        # INFERENCE
        else:
            # conditioned reverse diffusion process
            if self.inferenceInitialSampling == "random":
                dNoise = torch.randn_like(d, device=device)
                cNoise = torch.randn_like(cond, device=device)
            else:
                # Dynamic shape generation based on input dimensions
                # Create a shape tuple: (1, C, L) or (1, C, H, W)
                shape_d = (1,) + d.shape[1:]
                shape_c = (1,) + cond.shape[1:]
                
                # Expand dims: -1 for every dimension except batch
                expand_dims = [-1] * (d.ndim - 1)
                
                dNoise = torch.randn(shape_d, device=device).expand(d.shape[0], *expand_dims)
                cNoise = torch.randn(shape_c, device=device).expand(cond.shape[0], *expand_dims)

            sampleStride = 1

            if return_x0_estimate:
                all_x0_estimates = []

            for i in reversed(range(0, self.timesteps, sampleStride)):
                t = i * torch.ones(cond.shape[0], device=device).long()
                
                # Fetch schedule params
                sqrt_alphas = extract(self.sqrtAlphasCumprod, t)
                sqrt_one_minus_alphas = extract(self.sqrtOneMinusAlphasCumprod, t)
                sqrt_recip_alphas = extract(self.sqrtRecipAlphas, t)
                betas = extract(self.betas, t)
                sqrt_posterior_var = extract(self.sqrtPosteriorVariance, t)

                # compute conditioned part with normal forward diffusion
                if self.inferenceConditioningIntegration == "noisy":
                    condNoisy = sqrt_alphas * cond + sqrt_one_minus_alphas * cNoise
                else:
                    condNoisy = cond

                if input_type == "clean":
                    dNoise = torch.randn_like(d, device=device)
                    dNoise = sqrt_alphas * d + sqrt_one_minus_alphas * dNoise

                elif input_type == "prev-pred":

                    t_prev= torch.minimum(t + 1, torch.tensor(self.timesteps-1))
                    sqrt_alphas_prev = extract(self.sqrtAlphasCumprod, t_prev)
                    sqrt_one_minus_alphas_prev = extract(self.sqrtOneMinusAlphasCumprod, t_prev)

                    dNoise = torch.randn_like(d, device=device)
                    dNoise = sqrt_alphas_prev * d + sqrt_one_minus_alphas_prev * dNoise
                    
                    dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)
                    predictedNoiseCond = self.unet(dNoiseCond, t_prev)
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:]  - sqrt_one_minus_alphas_prev * predictedNoiseCond[:, cond.shape[1]:])/sqrt_alphas_prev

                    dNoise = torch.randn_like(d, device=device)
                    dNoise = sqrt_alphas * x0_estimate + sqrt_one_minus_alphas * dNoise  
                
                elif input_type == "own-pred":                
                    t_prev = t
                    # (Note: Reuse previously fetched scalars for t)

                    dNoise = torch.randn_like(d, device=device)
                    dNoise = sqrt_alphas * d + sqrt_one_minus_alphas * dNoise
                
                    dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)
                    predictedNoiseCond = self.unet(dNoiseCond, t_prev)
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:]  - sqrt_one_minus_alphas * predictedNoiseCond[:, cond.shape[1]:])/sqrt_alphas
                    
                    dNoise = torch.randn_like(d, device=device)
                    dNoise = sqrt_alphas * x0_estimate + sqrt_one_minus_alphas * dNoise
                    
                dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)

                # backward diffusion process that removes noise to create data
                predictedNoiseCond = self.unet(dNoiseCond, t)

                # use model (noise predictor) to predict mean
                modelMean = sqrt_recip_alphas * (dNoiseCond - betas * predictedNoiseCond / sqrt_one_minus_alphas)

                dNoise = modelMean[:, cond.shape[1]:modelMean.shape[1]] # discard prediction of conditioning
                if i != 0 and self.inferenceSamplingMode == "ddpm":
                    if self.inferencePosteriorSampling == "random":
                        # sample randomly (only for non-final prediction)
                        dNoise = dNoise + sqrt_posterior_var * torch.randn_like(dNoise)
                    else:
                        # sample with same seed (only for non-final prediction)
                        shape_dNoise = (1,) + dNoise.shape[1:]
                        expand_dims = [-1] * (dNoise.ndim - 1)
                        
                        postNoise = torch.randn(shape_dNoise, device=device).expand(dNoise.shape[0], *expand_dims)
                        dNoise = dNoise + sqrt_posterior_var * postNoise

                if return_x0_estimate:
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:modelMean.shape[1]]  - sqrt_one_minus_alphas * predictedNoiseCond[:, cond.shape[1]:modelMean.shape[1]])/sqrt_alphas
                    all_x0_estimates.append(x0_estimate)

            if return_x0_estimate:
                return dNoise, torch.stack(all_x0_estimates)
            return dNoise