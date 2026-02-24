import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.diffusion import compute_sigmas_refiner, prep
from .unet_2d import Unet
from .unet_acdm import UnetACDM
from .unet_1d import Unet1D


class PDERefiner(nn.Module):
    """
    PDERefiner: iterative PDE solution refinement via cold diffusion.

    Key differences from classical DiffusionModel:
    - At the maximum timestep T:
        Training  — noisy input is zeros (not Gaussian), target is x0 (not epsilon)
        Inference — cold start from zeros, model output is taken as x0 directly
    - At timesteps t < T:
        Training  — classical diffusion (noisy input, predict epsilon)
        Inference — re-noise the previous x0 estimate at level t, predict epsilon,
                    recover x0 via the standard formula
    """

    def __init__(self, dimension, dataSize, condChannels, dataChannels, refinementSteps, log_sigma_min,
                 padding_mode='circular', architecture="Unet2D", checkpoint=""):
        super(PDERefiner, self).__init__()

        self.dimension = dimension
        self.refinementSteps = refinementSteps
        self.condChannels = condChannels
        self.dataChannels = dataChannels
        self.sigma_min = 10**log_sigma_min
        self.architecture = architecture

        ''' if self.architecture == "ACDM":
            self.unet = UnetACDM(
                dim=128,
                channels=condChannels + dataChannels,
                sigmas=torch.zeros(1),
                dim_mults=(1, 1, 1),
                use_convnext=True,
                convnext_mult=1,
            )'''

        if self.dimension == 2:
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
            padding_mode=padding_mode
            )

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        if checkpoint != "":
            print(f"Loading Checkpoint from {checkpoint}")
            ckpt = torch.load(checkpoint)
            if 'stateDictDecoder' in ckpt.keys():
                ckpt = ckpt['stateDictDecoder']
            checkpoint_unet = {key[5:]: ckpt[key] for key in ckpt
                               if 'unet' in key and 'sigmas' not in key}
            if 'sigmas' not in checkpoint_unet:
                checkpoint_unet['sigmas'] = torch.tensor([1])
            self.unet.load_state_dict(checkpoint_unet)

        sigmas = compute_sigmas_refiner(self.sigma_min, refinementSteps=self.refinementSteps)
        sigmas = torch.cat((sigmas, torch.ones(1)))
        self.register_buffer("sigmas", prep(sigmas, self.dimension))
        print(self.sigmas)

        self.unet.sigmas = 1 / torch.ravel(self.sigmas)
        
        self.nTimesteps = len(self.sigmas)
        self.timesteps = torch.arange(self.nTimesteps)
        self.trainingTimesteps = torch.cat((torch.zeros(10), self.timesteps))
        
    def forward(self, conditioning: torch.Tensor, data: torch.Tensor = None,
                return_x0_estimate: bool = False, input_type: str = "ancestor") -> torch.Tensor:

        device = conditioning.device

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
            p = torch.ones(len(self.trainingTimesteps))/len(self.trainingTimesteps)
            index = p.multinomial(num_samples=d.shape[0], replacement=True)
            t = self.trainingTimesteps[index].long().to(device)
            dNoise = torch.randn_like(d)
            dNoisy = d + self.sigmas[t] * dNoise

            # At max timestep: replace noisy input with zeros (cold start)
            is_max_t = (t == self.nTimesteps - 1).view(-1, *([1] * (d.ndim - 1)))
            dNoisy = torch.where(is_max_t, torch.zeros_like(dNoisy), dNoisy)

            input_cat = torch.cat((cond, dNoisy), dim=1)
            predicted = self.unet(input_cat, t)[:, cond.shape[1]:]

            # At max timestep target is x0; otherwise epsilon
            target = torch.where(is_max_t, d, dNoise)

            return target, predicted

        # ==========================
        # INFERENCE
        # ==========================
        else:
            x0_estimate = torch.zeros_like(d)

            if return_x0_estimate:
                all_x0_estimates = []

            for i in reversed(range(self.nTimesteps)):
                t = torch.full((cond.shape[0],), i, device=device, dtype=torch.long)

                if i == self.nTimesteps - 1:
                    # Cold start at max timestep: all modes share this behaviour
                    dNoisy = torch.zeros_like(d)

                else:
                    # Optional: pre-processing of dNoisy based on input_type
                    if input_type == "clean":
                        # Re-noise reference d at the current sigma level
                        dNoisy = d + self.sigmas[t] * torch.randn_like(d)

                    elif input_type == "prev-pred":
                        # Look one step ahead (higher sigma = more noisy)
                        t_prev = torch.clamp(t + 1, max=self.nTimesteps - 1)

                        if i == self.nTimesteps - 2:
                            # The preceding step was the cold-start: mirror that with zeros
                            dNoisy_prev = torch.zeros_like(d)
                        else:
                            dNoisy_prev = d + self.sigmas[t_prev] * torch.randn_like(d)

                        input_cat_prev = torch.cat((cond, dNoisy_prev), dim=1)
                        predicted_prev = self.unet(input_cat_prev, t_prev)[:, cond.shape[1]:]

                        if i + 1 == self.nTimesteps - 1:
                            # At max t the model outputs x0 directly
                            x0_tmp = predicted_prev
                        else:
                            x0_tmp = dNoisy_prev - self.sigmas[t_prev] * predicted_prev

                        # Re-noise the lookahead x0 estimate to the current level
                        dNoisy = x0_tmp + self.sigmas[t] * torch.randn_like(d)

                    elif input_type == "own-pred":
                        # 1. Forward-noise d to the current sigma level
                        dNoisy = d + self.sigmas[t] * torch.randn_like(d)
                        # 2. Quick prediction to get a preliminary x0
                        input_cat_tmp = torch.cat((cond, dNoisy), dim=1)
                        predicted_tmp = self.unet(input_cat_tmp, t)[:, cond.shape[1]:]
                        x0_tmp = dNoisy - self.sigmas[t] * predicted_tmp
                        # 3. Re-noise the preliminary x0 at the current level
                        dNoisy = x0_tmp + self.sigmas[t] * torch.randn_like(d)

                    else:  # "ancestor" (default)
                        # Re-noise the previous x0 estimate at noise level t
                        dNoisy = x0_estimate + self.sigmas[t] * torch.randn_like(d)

                # --- Standard refinement step ---
                input_cat = torch.cat((cond, dNoisy), dim=1)
                predicted = self.unet(input_cat, t)[:, cond.shape[1]:]

                if i == self.nTimesteps - 1:
                    # At max timestep the model outputs x0 directly
                    x0_estimate = predicted
                else:
                    # Recover x0 from the predicted epsilon
                    x0_estimate = dNoisy - self.sigmas[t] * predicted

                if return_x0_estimate:
                    all_x0_estimates.append(x0_estimate)

            if return_x0_estimate:
                return x0_estimate, torch.stack(all_x0_estimates)
            return x0_estimate
