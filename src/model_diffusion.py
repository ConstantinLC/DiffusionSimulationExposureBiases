import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion_utils import linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from src.diffusion_utils import cosine_beta_schedule, cubic_beta_schedule, psd_beta_schedule, log_uniform_beta_schedule
from src.diffusion_utils import piecewise_log_beta_schedule
from src.diffusion_utils import betas_from_sqrtOneMinusAlphasCumprod
from src.diffusion_utils import ddim_x0_estimate
from src.model import Unet
from src.model_acdm_arch import UnetACDM

from matplotlib import pyplot as plt

### DIFFUSION MODEL WITH CONDITIONING
class DiffusionModel(nn.Module):

    def __init__(self, dimension, dataSize, condChannels, dataChannels, 
                 diffSchedule, diffSteps, inferenceSamplingMode, 
                 inferenceConditioningIntegration, diffCondIntegration, 
                 diffScheduleB = "linear",
                 inferenceInitialSampling = "random", x0_estimate_type="mean", padding_mode='circular',
                 scheduled_sampling = False, architecture="ours"):
        super(DiffusionModel, self).__init__()

        self.dimension = dimension
        self.timesteps = diffSteps
        
        # sampling settings
        self.inferencePosteriorSampling = "random"      # "random" or "same", ddpm only
        self.inferenceInitialSampling = inferenceInitialSampling       # "random" or "same"
        self.inferenceConditioningIntegration = inferenceConditioningIntegration # "noisy" or "clean"
        self.inferenceSamplingMode = inferenceSamplingMode # "ddpm" or "ddim"
        self.diffCondIntegration = diffCondIntegration # "noisy" or "clean"
        self.scheduled_sampling = scheduled_sampling
        self.architecture = architecture

        # --- SCHEDULE A (Physics/Sampling) ---
        betas = self._get_betas_from_schedule(diffSchedule)
        # Format betas for broadcasting
        betas = betas.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        # Standard calculations for Schedule A
        alphas = 1.0 - betas
        alphasCumprod = torch.cumprod(alphas, axis=0)
        # Pad with 1.0 at the beginning for t=0 calculation
        alphasCumprodPrev = F.pad(alphasCumprod[:-1], (0,0,0,0,0,0,1,0), value=1.0)
        sqrtRecipAlphas = torch.sqrt(1.0 / alphas)
        sqrtAlphasCumprod = torch.sqrt(alphasCumprod)
        sqrtOneMinusAlphasCumprod = torch.sqrt(1. - alphasCumprod)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posteriorVariance = betas * (1. - alphasCumprodPrev) / (1. - alphasCumprod)
        sqrtPosteriorVariance = torch.sqrt(posteriorVariance)

        # Register Schedule A buffers
        self.register_buffer("betas", betas)
        self.register_buffer("sqrtRecipAlphas", sqrtRecipAlphas)
        self.register_buffer("sqrtAlphasCumprod", sqrtAlphasCumprod)
        self.register_buffer("sqrtOneMinusAlphasCumprod", sqrtOneMinusAlphasCumprod)
        self.register_buffer("sqrtPosteriorVariance", sqrtPosteriorVariance)

        # --- WEIGHT CALCULATION (Schedule A vs Schedule B) ---
        # 1. Calculate sqrtAlphasCumprod for A (already done above)
        sqrtAlphasCumprodPrev_A = torch.sqrt(alphasCumprodPrev)
        
        # 2. Calculate sqrtAlphasCumprod for B
        betas_B = self._get_betas_from_schedule(diffScheduleB).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        alphas_B = 1.0 - betas_B
        alphasCumprod_B = torch.cumprod(alphas_B, axis=0)
        alphasCumprodPrev_B = F.pad(alphasCumprod_B[:-1], (0,0,0,0,0,0,1,0), value=1.0)
        
        sqrtAlphasCumprod_B = torch.sqrt(alphasCumprod_B)
        sqrtAlphasCumprodPrev_B = torch.sqrt(alphasCumprodPrev_B)

        # 3. Calculate Deltas (Derivatives of sqrtAlpha w.r.t steps)
        # We add epsilon to denominator to prevent division by zero in pathological schedules
        epsilon = 1e-8
        delta_A = sqrtAlphasCumprod - sqrtAlphasCumprodPrev_A
        delta_B = sqrtAlphasCumprod_B - sqrtAlphasCumprodPrev_B

        # 4. Compute Weights: Ratio of slope A to slope B
        # If A is steep (delta large), we sample it rarely (in time). To match B (if B is flat/delta small),
        # we need to upweight A. 
        # Importance Weight = P_target(x) / P_source(x) 
        # P(x) ~ 1 / |delta|. Therefore Weight ~ |delta_A| / |delta_B|
        step_weights = torch.abs(delta_A) / (torch.abs(delta_B) + epsilon)
        fig = plt.figure()
        plt.plot(sqrtAlphasCumprod.ravel(), step_weights.ravel())
        plt.hist(sqrtAlphasCumprod.ravel(), alpha=0.2)
        plt.hist(sqrtAlphasCumprod_B.ravel(), alpha=0.2)
        fig.savefig("/mnt/SSD2/constantin/diffusion-multisteps/results/weighting.png")
        
        #self.register_buffer("step_weights", step_weights)
        self.step_weights = step_weights.to('cuda')
        print(step_weights)

        self.condChannels = condChannels
        self.dataChannels = dataChannels

        if self.architecture == "ACDM":
            self.unet = UnetACDM(dim=128,
                channels= condChannels + dataChannels,
                dim_mults=(1,1,1),
                use_convnext=True,
                convnext_mult=1)
        
        elif self.architecture == "ours":
            self.unet = Unet(
            dim=dataSize[0],
            channels=condChannels+dataChannels,
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
            padding_mode=padding_mode
            )
        
        else : raise Exception

        self.x0_estimate_type = x0_estimate_type

    def _get_betas_from_schedule(self, schedule_name):
        if schedule_name == "linear":
            betas = linear_beta_schedule(timesteps=self.timesteps)
        elif schedule_name == "quadratic":
            betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif schedule_name == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        elif schedule_name == "cosine":
            betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif schedule_name == "cubic":
            betas = cubic_beta_schedule(timesteps=self.timesteps)
        elif schedule_name == "psd":
            betas = psd_beta_schedule(timesteps=self.timesteps)
        elif schedule_name == "log-minus5-uniform":
            betas = log_uniform_beta_schedule(timesteps=self.timesteps, min=-5)
        elif schedule_name == "log-minus7-uniform":
            betas = log_uniform_beta_schedule(timesteps=self.timesteps, min=-7)
        elif schedule_name == "piecewise-log":
            betas = piecewise_log_beta_schedule(timesteps=self.timesteps)
        elif schedule_name == "PSD++":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.7, -1.5, 40),
                                                                            torch.logspace(-1.5, -0.1, 45),
                                                                            torch.logspace(-0.1, -0.0005, 15))))
        elif schedule_name == "PSD+":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.7, -1.5, 30),
                                                                            torch.logspace(-1.5, -0.1, 55),
                                                                            torch.logspace(-0.1, -0.0001, 15))))
        elif schedule_name == "PSD+++":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.7, -1.4, 50),
                                                                            torch.logspace(-1.5, -0.1, 35),
                                                                            torch.logspace(-0.1, -0.0005, 15))))
        elif schedule_name == "PSD+MinLogMinus2":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.5, 30),
                                                                            torch.logspace(-1.5, -0.1, 55),
                                                                            torch.logspace(-0.1, -0.0005, 15))))
        else:
            raise ValueError(f"Unknown variance schedule: {schedule_name}")
        return betas

    # input shape (both inputs): B S C W H (D) -> output shape (both outputs): B S nC W H (D)
    def forward(self, conditioning:torch.Tensor, data:torch.Tensor = None, return_x0_estimate:bool = False, 
                return_denoiser_inputs:bool = False, input_type:str = "ancestor", lower_limit_timesteps_training:int = None, upper_limit_timesteps_training:int = None, 
                error_amount:float = None, correcting_unet:nn.Module = None) -> torch.Tensor:

        device = "cuda" if conditioning.is_cuda else "cpu"

        if data is None:
            N, C_cond, H, W = conditioning.shape
            data = torch.zeros((N, self.dataChannels, H, W)).to(device)
        if self.dimension == 3:
            raise NotImplementedError()
        # combine batch and sequence dimension for decoder processing
        d = data
        cond = conditioning

        if lower_limit_timesteps_training is None:
            lower_limit_timesteps_training = 0

        if upper_limit_timesteps_training is None:
            upper_limit_timesteps_training = self.timesteps

        # TRAINING
        if self.training:

            original_d = d
            t = torch.randint(lower_limit_timesteps_training, upper_limit_timesteps_training, (d.shape[0],), device=device).long()

            if self.scheduled_sampling : 
                t_prev = torch.minimum(t + 1, torch.ones(t.shape).long().to(device)*(self.timesteps-1))

                # forward diffusion process that adds noise to data (Uses Schedule A)
                if self.diffCondIntegration == "noisy":
                    d = torch.concat((cond, d), dim=1)
                    noise = torch.randn_like(d, device=device)
                    dNoisy = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * noise

                elif self.diffCondIntegration == "clean":
                    dNoise = torch.randn_like(d, device=device)
                    dNoisy = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * dNoise

                    noise = torch.concat((cond, dNoise), dim=1)
                    dNoisy = torch.concat((cond, dNoisy), dim=1)
                
                predictedNoise = self.unet(dNoisy, t_prev)[:, cond.shape[1]:]
                d = (dNoisy[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t_prev] * predictedNoise)/self.sqrtAlphasCumprod[t_prev]

            # forward diffusion process that adds noise to data (Uses Schedule A)
            if self.diffCondIntegration == "noisy":
                d = torch.concat((cond, d), dim=1)
                noise = torch.randn_like(d, device=device)
                dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * noise

            elif self.diffCondIntegration == "clean":
                dNoise = torch.randn_like(d, device=device)
                dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * dNoise

                noise = torch.concat((cond, dNoise), dim=1)
                dNoisy = torch.concat((cond, dNoisy), dim=1)

            else:
                raise ValueError("Unknown conditioning integration mode")

            # noise prediction with network
            predictedNoise = self.unet(dNoisy, t)[:, cond.shape[1]:]
            noise = noise[:, cond.shape[1]:]

            # Retrieve pre-calculated importance weights
            weights = self.step_weights[t]
            scale_factor = torch.sqrt(weights).view(-1, 1, 1, 1)

            if return_x0_estimate:
                if self.x0_estimate_type == "mean":
                    x0_estimate = (dNoisy[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t] * predictedNoise)/self.sqrtAlphasCumprod[t]
                elif self.x0_estimate_type == "sample":
                    x0_estimate = ddim_x0_estimate(dNoisy, t, self, self.unet, 0, 0)
                return noise, predictedNoise, x0_estimate, weights
                        
            noise_scaled = noise * scale_factor
            predictedNoise_scaled = predictedNoise * scale_factor
            return noise_scaled, predictedNoise_scaled

        # INFERENCE
        else:
            # Inference uses Standard Schedule (A) buffers
            #torch.manual_seed(1)
            #torch.cuda.manual_seed(1)

            # conditioned reverse diffusion process
            if self.inferenceInitialSampling == "random":
                dNoise = torch.randn_like(d, device=device)
                cNoise = torch.randn_like(cond, device=device)
            else:
                dNoise = torch.randn((1, d.shape[1], d.shape[2], d.shape[3]), device=device).expand(d.shape[0],-1,-1,-1)
                cNoise = torch.randn((1, cond.shape[1], cond.shape[2], cond.shape[3]), device=device).expand(cond.shape[0],-1,-1,-1)

            sampleStride = 1

            if return_x0_estimate:
                all_x0_estimates = []

            if return_denoiser_inputs:
                denoiser_inputs = []

            for i in reversed(range(0, self.timesteps, sampleStride)):
                t = i * torch.ones(cond.shape[0], device=device).long()

                # compute conditioned part with normal forward diffusion
                if self.inferenceConditioningIntegration == "noisy":
                    condNoisy = self.sqrtAlphasCumprod[t] * cond + self.sqrtOneMinusAlphasCumprod[t] * cNoise
                else:
                    condNoisy = cond

                if torch.sum(t) == 0 and correcting_unet is not None:
                    condNoisy = condNoisy + correcting_unet(condNoisy, time=None)

                if input_type == "clean":
                    dNoise = torch.randn_like(d, device=device)
                    dNoise = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * dNoise

                elif "clean-previous" in  input_type:
                    nb_previous = int(input_type.split("-")[2])
                
                    for i_previous in reversed(range(1, nb_previous+1, 1)):
                        t_prev= torch.minimum(t + i_previous, torch.tensor(self.timesteps-1))

                        if i_previous == nb_previous:
                            dNoise = torch.randn_like(d, device=device)
                            dNoise = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * dNoise
                        
                        dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)
                        predictedNoiseCond = self.unet(dNoiseCond, t_prev)
                        modelMean = self.sqrtRecipAlphas[t_prev] * (dNoiseCond - self.betas[t_prev] * predictedNoiseCond / self.sqrtOneMinusAlphasCumprod[t_prev])

                        dNoise = modelMean[:, cond.shape[1]:modelMean.shape[1]]
                        dNoise = dNoise + self.sqrtPosteriorVariance[t_prev] * torch.randn_like(dNoise)
                    
                dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)

                if return_denoiser_inputs:
                    denoiser_inputs.append(dNoise)

                # backward diffusion process that removes noise to create data
                predictedNoiseCond = self.unet(dNoiseCond, t)

                # use model (noise predictor) to predict mean
                modelMean = self.sqrtRecipAlphas[t] * (dNoiseCond - self.betas[t] * predictedNoiseCond / self.sqrtOneMinusAlphasCumprod[t])

                dNoise = modelMean[:, cond.shape[1]:modelMean.shape[1]] # discard prediction of conditioning
                if i != 0 and self.inferenceSamplingMode == "ddpm":
                    if self.inferencePosteriorSampling == "random":
                        # sample randomly (only for non-final prediction)
                        dNoise = dNoise + self.sqrtPosteriorVariance[t] * torch.randn_like(dNoise)
                    else:
                        # sample with same seed (only for non-final prediction)
                        postNoise = torch.randn((1, dNoise.shape[1], dNoise.shape[2], dNoise.shape[3]), device=device).expand(dNoise.shape[0],-1,-1,-1)
                        dNoise = dNoise + self.sqrtPosteriorVariance[t] * postNoise

                if return_x0_estimate:
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:modelMean.shape[1]]  - self.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:modelMean.shape[1]])/self.sqrtAlphasCumprod[t]
                    all_x0_estimates.append(x0_estimate)

            if return_x0_estimate:
                if return_denoiser_inputs:
                    return dNoise, torch.stack(all_x0_estimates), torch.stack((denoiser_inputs))
                return dNoise, torch.stack(all_x0_estimates)
            return dNoise