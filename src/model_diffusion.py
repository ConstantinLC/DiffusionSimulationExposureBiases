### Mainly copied from https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion.py

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

### DIFFUSION MODEL WITH CONDITIONING
class DiffusionModel(nn.Module):

    def __init__(self, dimension, dataSize, condChannels, dataChannels, diffSchedule, diffSteps, 
                 inferenceSamplingMode, inferenceConditioningIntegration, diffCondIntegration, 
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
        print(diffSchedule)
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
        elif diffSchedule == "log-minus5-uniform":
            betas = log_uniform_beta_schedule(timesteps=self.timesteps, min=-5)
        elif diffSchedule == "log-minus7-uniform":
            betas = log_uniform_beta_schedule(timesteps=self.timesteps, min=-7)
        elif diffSchedule == "piecewise-log":
            betas = piecewise_log_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "PSD++":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.7, -1.5, 40),
                                                                            torch.logspace(-1.5, -0.1, 45),
                                                                            torch.logspace(-0.1, -0.0005, 15))))
        elif diffSchedule == "PSD+":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.7, -1.5, 30),
                                                                            torch.logspace(-1.5, -0.1, 55),
                                                                            torch.logspace(-0.1, -0.0001, 15))))
        elif diffSchedule == "PSD+++":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.7, -1.4, 50),
                                                                            torch.logspace(-1.5, -0.1, 35),
                                                                            torch.logspace(-0.1, -0.0005, 15))))
        elif diffSchedule == "PSD+MinLogMinus2":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.5, 30),
                                                                            torch.logspace(-1.5, -0.1, 55),
                                                                            torch.logspace(-0.1, -0.0005, 15))))
        elif diffSchedule == "PSD-MoreLeftSpectrum-Kolmo":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.67, -1.5, 60),
                            torch.logspace(-1.5, -0.1, 30),
                            torch.logspace(-0.1, -0.0001, 10))))
        elif diffSchedule == "PSD-MoreMoreLeftSpectrum-Kolmo":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.67, -1.5, 85),
                            torch.logspace(-1.5, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
        elif diffSchedule == "Log-1.8-Min":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.8, -1.5, 85),
                            torch.logspace(-1.5, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
        elif diffSchedule == "Log-2-Min":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.8, 85),
                            torch.logspace(-1.8, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
        elif diffSchedule == "Log-2":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.logspace(-2, -1.99, 100))
        elif diffSchedule == "Log-1.6-Min":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.6, -1.4, 85),
                            torch.logspace(-1.4, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
        else:
            raise ValueError("Unknown variance schedule")
        
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

        self.register_buffer("betas", betas)
        self.register_buffer("sqrtRecipAlphas", sqrtRecipAlphas)
        self.register_buffer("sqrtAlphasCumprod", sqrtAlphasCumprod)
        self.register_buffer("sqrtOneMinusAlphasCumprod", sqrtOneMinusAlphasCumprod)
        self.register_buffer("sqrtPosteriorVariance", sqrtPosteriorVariance)

        self.condChannels = condChannels
        self.dataChannels = dataChannels
        print(diffSchedule)
        if self.architecture == "ACDM":
            self.unet = UnetACDM(dim=128,
                channels= condChannels + dataChannels,
                sigmas = sqrtAlphasCumprod/sqrtOneMinusAlphasCumprod,
                dim_mults=(1,1,1),
                use_convnext=True,
                convnext_mult=1)
        
        elif self.architecture == "ours":
            self.unet = Unet(
            dim=dataSize[0],
            sigmas = self.sqrtAlphasCumprod/self.sqrtOneMinusAlphasCumprod,
            channels=condChannels+dataChannels,
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
            padding_mode=padding_mode
            )
        
        else : raise Exception

        self.x0_estimate_type = x0_estimate_type

    def resetSchedule(self, diffSchedule):
        if diffSchedule == "linear":
            betas = linear_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "cosine":
            betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "psd":
            betas = psd_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "psd+":
            target = torch.concatenate((torch.logspace(-1.7, -1.5, 30),
                            torch.logspace(-1.5, -0.1, 55),
                            torch.logspace(-0.1, -0.0001, 15)))
            
            if self.timesteps == 20:
                target = target[4::5]

            betas = betas_from_sqrtOneMinusAlphasCumprod(target)
        elif diffSchedule == "linear-raccourci":
            target = torch.tensor([0.1764, 0.2500, 0.2903, 0.3981, 0.4987, 0.5907, 0.6730, 0.7447, 0.8056,
                                    0.8559, 0.8962, 0.9276, 0.9511, 0.9682, 0.9800, 0.9880, 0.9931, 0.9962,
                                    0.9980, 0.9990])
            betas = betas_from_sqrtOneMinusAlphasCumprod(target)
        elif diffSchedule == "transonic_iteration0":
            target = torch.cat((torch.logspace(-0.8, -0.2, 10), torch.logspace(-0.2, -0.0005, 10)))
            betas = betas_from_sqrtOneMinusAlphasCumprod(target)
        elif diffSchedule == "transonic_iteration1":
            target = torch.tensor([0.3981, 0.4484, 0.4987, 0.5447, 0.5907, 0.6318, 0.6730, 0.7447, 0.8056,
                            0.8559, 0.8962, 0.9276, 0.9511, 0.9682, 0.9800, 0.9880, 0.9931, 0.9962,
                            0.9980, 0.9990])
            betas = betas_from_sqrtOneMinusAlphasCumprod(target)
        elif diffSchedule == "transonic_iteration2":
            target = torch.cat((torch.logspace(-0.7, -0.2, 14), torch.logspace(-0.17, -0.0005, 6)))
            betas = betas_from_sqrtOneMinusAlphasCumprod(target)
        elif diffSchedule == "transonic_iteration3":
            target = torch.cat((torch.logspace(-0.7, -0.2, 14), torch.logspace(-0.16, -0.0005, 6)))
            betas = betas_from_sqrtOneMinusAlphasCumprod(target)
        elif diffSchedule == "transonic_iteration3_fullynoisefirststep":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.cat((torch.logspace(-0.7, -0.2, 14), torch.logspace(-0.16, -0.00005, 6))))
        elif diffSchedule == "transonic_iteration4":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.cat((torch.logspace(-0.7, -0.5, 10), torch.logspace(-0.45, -0.0005, 10))))
        elif diffSchedule == "transonic_iteration5":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.cat((torch.logspace(-1, -0.6, 10), torch.logspace(-0.55, -0.00005, 10))))
        elif diffSchedule == "transonic_iteration6":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.cat((torch.logspace(-1.2, -1, 10), torch.logspace(-0.55, -0.00005, 10))))
        elif diffSchedule == "transonic_iteration7":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.cat((torch.logspace(-1.2, -1, 14), torch.logspace(-0.16, -0.00005, 6))))
        elif diffSchedule == "PSD-MoreLeftSpectrum-Kolmo":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.67, -1.5, 60),
                            torch.logspace(-1.5, -0.1, 30),
                            torch.logspace(-0.1, -0.0001, 10))))
        elif diffSchedule == "PSD-MoreMoreLeftSpectrum-Kolmo":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.67, -1.5, 85),
                            torch.logspace(-1.5, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
        elif diffSchedule == "Log-1.8-Min":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.8, -1.5, 85),
                            torch.logspace(-1.5, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
        elif diffSchedule == "Log-2-Min":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.8, 85),
                            torch.logspace(-1.8, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
        elif diffSchedule == "Log-2":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.logspace(-2, -1.99, 100))
        elif diffSchedule == "Log-1.6-Min":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.6, -1.4, 85),
                            torch.logspace(-1.4, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
        else:
            raise ValueError("Unknown variance schedule")
    
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


    # input shape (both inputs): B S C W H (D) -> output shape (both outputs): B S nC W H (D)
    def forward(self, conditioning:torch.Tensor, data:torch.Tensor = None, return_x0_estimate:bool = False, 
                return_denoiser_inputs:bool = False, input_type:str = "ancestor", lower_limit_timesteps_training:int = None, upper_limit_timesteps_training:int = None, 
                error_amount:float = None,correcting_unet:nn.Module = None) -> torch.Tensor:

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

            if "input-own-pred" in input_type:    
                dNoise = torch.randn_like(d, device=device)
                dNoise = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * dNoise
            
                dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)
                predictedNoiseCond = self.unet(dNoiseCond, t_prev)
                modelMean = self.sqrtRecipAlphas[t] * (dNoiseCond - self.betas[t] * predictedNoiseCond / self.sqrtOneMinusAlphasCumprod[t])

                dNoise = modelMean[:, cond.shape[1]:modelMean.shape[1]]
                dNoise = dNoise + self.sqrtPosteriorVariance[t] * torch.randn_like(dNoise)

                # forward diffusion process that adds noise to data
                if self.diffCondIntegration == "noisy":
                    d = torch.concat((cond, d), dim=1)
                    noise = torch.randn_like(d, device=device)
                    dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * noise

                elif self.diffCondIntegration == "clean":
                    dNoise = torch.randn_like(d, device=device)
                    dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * dNoise

                    noise = torch.concat((cond, dNoise), dim=1)
                    dNoisy = torch.concat((cond, dNoisy), dim=1)
                
                predictedNoise_own_pred = self.unet(dNoisy, t)[:, cond.shape[1]:]
                noise_own_pred = noise[:, cond.shape[1]:]
                d = (dNoisy[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t] * predictedNoise)/self.sqrtAlphasCumprod[t]

            # forward diffusion process that adds noise to data
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
            # unstack batch and sequence dimension again

            if self.scheduled_sampling:
                return noise, predictedNoise, noise_own_pred, predictedNoise_own_pred

            if return_x0_estimate:
                x0_estimate = (dNoisy[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t] * predictedNoise)/self.sqrtAlphasCumprod[t]
                return noise, predictedNoise, x0_estimate
                        
            return noise, predictedNoise


        # INFERENCE
        else:
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

                elif "clean-previous" in input_type:
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

                elif "input-own-pred" in input_type:    
                    t_prev= t

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

                """if i == self.limit_timesteps_training and i != 0:
                    dNoise = (dNoiseCond[:, -self.dimension:]  - self.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, -self.dimension:])/self.sqrtAlphasCumprod[t]
                    break"""

            if return_x0_estimate:
                if return_denoiser_inputs:
                    return dNoise, torch.stack(all_x0_estimates), torch.stack((denoiser_inputs))
                return dNoise, torch.stack(all_x0_estimates)
            return dNoise