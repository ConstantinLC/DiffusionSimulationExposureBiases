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
            
        elif diffSchedule == "second_iter":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.tensor([0.01259869709610939,
        0.01259869709610939,
        0.012601062655448914,
        0.012605791911482811,
        0.012608155608177185,
        0.012612882070243359,
        0.012617606669664383,
        0.012619968503713608,
        0.012624691240489483,
        0.012627051211893559,
        0.01263177115470171,
        0.01263413019478321,
        0.012638846412301064,
        0.012643561698496342,
        0.012645918875932693,
        0.012645918875932693,
        0.012650631368160248,
        0.01265298668295145,
        0.01265769638121128,
        0.012662405148148537,
        0.01266475860029459,
        0.012669463641941547,
        0.012671815231442451,
        0.01267651841044426,
        0.012678869068622589,
        0.012683569453656673,
        0.012688267976045609,
        0.012690616771578789,
        0.0126953125,
        0.0126953125,
        0.012697659432888031,
        0.012702353298664093,
        0.012707044370472431,
        0.012709389440715313,
        0.012714078649878502,
        0.012716422788798809,
        0.012721109203994274,
        0.01272579375654459,
        0.012728135101497173,
        0.012732816860079765,
        0.012735157273709774,
        0.012739837169647217,
        0.012742175720632076,
        0.012746852822601795,
        0.012751528061926365,
        0.012753864750266075,
        0.012753864750266075,
        0.012758538126945496,
        0.012760872952640057,
        0.012765543535351753,
        0.012770211324095726,
        0.012772545218467712,
        0.012777211144566536,
        0.012779543176293373,
        0.012784206308424473,
        0.012786537408828735,
        0.012791197746992111,
        0.012795857153832912,
        0.012798186391592026,
        0.012802842073142529,
        0.012805170379579067,
        0.01280982419848442,
        0.012814476154744625,
        0.01281680166721344,
        0.01282145082950592,
        0.01282145082950592,
        0.01282377541065216,
        0.012828422710299492,
        0.012830745428800583,
        0.01283538993448019,
        0.012840032577514648,
        0.01284235343337059,
        0.0128469942137599,
        0.012849314138293266,
        0.012853952124714851,
        0.012858588248491287,
        0.01286090537905693,
        0.012865539640188217,
        0.012867855839431286,
        0.012872486375272274,
        0.012874801643192768,
        0.012879430316388607,
        0.012884057126939297,
        0.012884057126939297,
        0.016820916905999184,
        0.016820916905999184,
        0.02705448307096958,
        0.04373593628406525,
        0.07084622979164124,
        0.11484682559967041,
        0.1862279325723648,
        0.3020063042640686,
        0.4897845983505249,
        0.7943300008773804,
        0.7943300008773804,
        0.8413479328155518,
        0.8911492228507996,
        0.9438982605934143,
        0.9718340635299683,
        0.9997697472572327]))

        
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
        
        elif self.architecture == "ours":
            self.unet = Unet(
            dim=dataSize[0],
            sigmas=torch.zeros(1),
            channels=condChannels+dataChannels,
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
            padding_mode=padding_mode
            )
        
        else : raise Exception

        self.compute_schedule_variables(betas)

    def compute_schedule_variables(self, betas):
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

    # input shape (both inputs): B S C W H (D) -> output shape (both outputs): B S nC W H (D)
    def forward(self, conditioning:torch.Tensor, data:torch.Tensor = None, return_x0_estimate:bool = False, 
                return_denoiser_inputs:bool = False, input_type:str = "ancestor", 
                fixed_timestep:int = None) -> torch.Tensor:

        device = "cuda" if conditioning.is_cuda else "cpu"

        if data is None:
            N, C_cond, H, W = conditioning.shape
            data = torch.zeros((N, self.dataChannels, H, W)).to(device)
        if self.dimension == 3:
            raise NotImplementedError()
        # combine batch and sequence dimension for decoder processing
        d = data
        cond = conditioning

        # TRAINING
        if self.training:

            if fixed_timestep is None:
                lower_limit_timesteps = 0
                upper_limit_timesteps = self.timesteps

            else:
                lower_limit_timesteps = fixed_timestep    
                upper_limit_timesteps = fixed_timestep + 1

            t = torch.randint(lower_limit_timesteps, upper_limit_timesteps, (d.shape[0],), device=device).long()

            if "input-own-pred" in input_type:
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
            
                predictedNoiseCond = self.unet(dNoisy, t)
                predictedNoise = predictedNoiseCond[:, cond.shape[1]:]

                first_estimate = (dNoisy[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t] * predictedNoise)/self.sqrtAlphasCumprod[t]
                d = first_estimate
            
            elif "input-prev-pred" in input_type:
                # forward diffusion process that adds noise to data
                t_prev = t+1

                if self.diffCondIntegration == "noisy":
                    d = torch.concat((cond, d), dim=1)
                    noise = torch.randn_like(d, device=device)
                    dNoisy = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * noise

                elif self.diffCondIntegration == "clean":
                    dNoise = torch.randn_like(d, device=device)
                    dNoisy = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * dNoise

                    noise = torch.concat((cond, dNoise), dim=1)
                    dNoisy = torch.concat((cond, dNoisy), dim=1)
            
                predictedNoiseCond = self.unet(dNoisy, t_prev)
                predictedNoise = predictedNoiseCond[:, cond.shape[1]:]

                first_estimate = (dNoisy[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t_prev] * predictedNoise)/self.sqrtAlphasCumprod[t_prev]
                d = first_estimate

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

            weights = self.weights[t]
            scale_factor = torch.sqrt(weights).view(-1, 1, 1, 1)

            if return_x0_estimate:
                x0_estimate = (dNoisy[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t] * predictedNoise)/self.sqrtAlphasCumprod[t]
                return x0_estimate
                        
            return scale_factor*noise, scale_factor*predictedNoise


        # INFERENCE
        else:

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

            for i in reversed(range(0, self.timesteps, sampleStride)):
                t = i * torch.ones(cond.shape[0], device=device).long()

                # compute conditioned part with normal forward diffusion
                if self.inferenceConditioningIntegration == "noisy":
                    condNoisy = self.sqrtAlphasCumprod[t] * cond + self.sqrtOneMinusAlphasCumprod[t] * cNoise
                else:
                    condNoisy = cond

                if input_type == "clean":
                    dNoise = torch.randn_like(d, device=device)
                    dNoise = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * dNoise

                elif input_type == "prev-pred":

                    t_prev= torch.minimum(t + 1, torch.tensor(self.timesteps-1))

                    dNoise = torch.randn_like(d, device=device)
                    dNoise = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * dNoise
                    
                    dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)
                    predictedNoiseCond = self.unet(dNoiseCond, t_prev)
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t_prev] * predictedNoiseCond[:, cond.shape[1]:])/self.sqrtAlphasCumprod[t_prev]

                    dNoise = torch.randn_like(d, device=device)
                    dNoise = self.sqrtAlphasCumprod[t] * x0_estimate + self.sqrtOneMinusAlphasCumprod[t] * dNoise  
                
                elif input_type == "own-pred":                
                    t_prev = t

                    dNoise = torch.randn_like(d, device=device)
                    dNoise = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * dNoise
                
                    dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)
                    predictedNoiseCond = self.unet(dNoiseCond, t_prev)
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:]  - self.sqrtOneMinusAlphasCumprod[t_prev] * predictedNoiseCond[:, cond.shape[1]:])/self.sqrtAlphasCumprod[t_prev]
                    
                    dNoise = torch.randn_like(d, device=device)
                    dNoise = self.sqrtAlphasCumprod[t] * x0_estimate + self.sqrtOneMinusAlphasCumprod[t] * dNoise
                    
                dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)

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
                return dNoise, torch.stack(all_x0_estimates)
            return dNoise