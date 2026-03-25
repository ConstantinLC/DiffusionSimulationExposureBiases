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

    multi_unet=True uses a separate Unet for each noise level instead of one shared Unet.
    """

    def __init__(self, dimension, dataSize, condChannels, dataChannels, refinementSteps, log_sigma_min,
                 padding_mode='circular', architecture="Unet2D", checkpoint="", multi_unet=False):
        super(PDERefiner, self).__init__()

        self.dimension = dimension
        self.refinementSteps = refinementSteps
        self.condChannels = condChannels
        self.dataChannels = dataChannels
        self.sigma_min = 10**log_sigma_min
        self.architecture = architecture
        self.multi_unet = multi_unet
        print(self.multi_unet)
        # ── Compute sigmas first (needed for nTimesteps when multi_unet=True) ──
        sigmas = compute_sigmas_refiner(self.sigma_min, refinementSteps=self.refinementSteps)
        sigmas = torch.cat((sigmas, torch.ones(1)))
        self.register_buffer("sigmas", prep(sigmas, self.dimension))
        print(self.sigmas)

        self.nTimesteps = len(self.sigmas)
        self.timesteps = torch.arange(self.nTimesteps)
        self.trainingTimesteps = torch.cat((torch.zeros(10), self.timesteps))

        # ── Create Unet(s) ─────────────────────────────────────────────────────
        def _make_unet():
            if architecture == "ACDM":
                if dimension != 2:
                    raise ValueError("ACDM architecture is only supported for dimension=2")
                return UnetACDM(
                    dim=dataSize[0],
                    sigmas=torch.zeros(1),
                    channels=condChannels + dataChannels,
                    dim_mults=(1, 1, 1),
                    use_convnext=True,
                    convnext_mult=1,
                )
            elif dimension == 2:
                return Unet(
                    dim=dataSize[0],
                    sigmas=torch.zeros(1),
                    channels=condChannels + dataChannels,
                    dim_mults=(1, 1, 1),
                    use_convnext=True,
                    convnext_mult=1,
                    padding_mode=padding_mode,
                )
            elif dimension == 1:
                return Unet1D(
                    dim=dataSize[0],
                    sigmas=torch.zeros(1),
                    channels=condChannels + dataChannels,
                    dim_mults=(1, 1, 1),
                    convnext_mult=1,
                    padding_mode=padding_mode,
                )
            else:
                raise ValueError(f"Unsupported dimension: {dimension}")

        if multi_unet:
            self.unets = nn.ModuleList([_make_unet() for _ in range(self.nTimesteps)])
        else:
            self.unet = _make_unet()

        # ── Load checkpoint ────────────────────────────────────────────────────
        if checkpoint != "":
            print(f"Loading Checkpoint from {checkpoint}")
            ckpt = torch.load(checkpoint)
            if 'stateDictDecoder' in ckpt.keys():
                ckpt = ckpt['stateDictDecoder']
            # Works for checkpoints saved from a single-unet PDERefiner
            checkpoint_unet = {key[5:]: ckpt[key] for key in ckpt
                               if 'unet' in key and 'sigmas' not in key}
            if 'sigmas' not in checkpoint_unet:
                checkpoint_unet['sigmas'] = torch.tensor([1])
            if multi_unet:
                for unet in self.unets:
                    unet.load_state_dict(checkpoint_unet)
            else:
                self.unet.load_state_dict(checkpoint_unet)

        # ── Set sigma embeddings on Unet(s) ───────────────────────────────────
        sigma_vals = 1 / torch.ravel(self.sigmas)
        if multi_unet:
            for unet in self.unets:
                unet.sigmas = sigma_vals
        else:
            self.unet.sigmas = sigma_vals

    # ── Unet dispatch ──────────────────────────────────────────────────────────

    def _apply_unet(self, inp: torch.Tensor, t: torch.Tensor,
                    step_idx: int = None) -> torch.Tensor:
        """
        Apply the appropriate Unet to inp.

        - single unet (multi_unet=False): always uses self.unet.
        - multi unet, inference (step_idx given): uses self.unets[step_idx].
        - multi unet, training (step_idx=None): groups the batch by t value and
          routes each group through self.unets[i].
        """
        if not self.multi_unet:
            return self.unet(inp, t)

        if step_idx is not None:
            return self.unets[step_idx](inp, t)

        # Training: batch may have mixed timestep values
        out = torch.empty_like(inp)
        for i in range(self.nTimesteps):
            mask = (t == i)
            if mask.any():
                out[mask] = self.unets[i](inp[mask], t[mask])
        return out

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, conditioning: torch.Tensor, data: torch.Tensor = None,
                return_x0_estimate: bool = False, input_type: str = "ancestor",
                return_noise_pred: bool = False, own_pred_iters: int = 1) -> torch.Tensor:

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
            predicted = self._apply_unet(input_cat, t)[:, cond.shape[1]:]

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

            if return_noise_pred:
                predictions = []
                targets = []

            for i in reversed(range(self.nTimesteps)):
                t = torch.full((cond.shape[0],), i, device=device, dtype=torch.long)

                if i == self.nTimesteps - 1:
                    # Cold start at max timestep: all modes share this
                    dNoisy = torch.zeros_like(d)
                    if return_noise_pred:
                        targets.append(d)

                else:
                    # Optional: pre-processing of dNoisy based on input_type
                    if input_type == "clean":
                        # Re-noise reference d at the current sigma level
                        dNoise = torch.randn_like(d)
                        dNoisy = d + self.sigmas[t] * dNoise

                    elif input_type == "prev-pred":
                        # Look one step ahead (higher sigma = more noisy)
                        t_prev = torch.clamp(t + 1, max=self.nTimesteps - 1)

                        if i == self.nTimesteps - 2:
                            # The preceding step was the cold-start: mirror that with zeros
                            dNoisy_prev = torch.zeros_like(d)
                        else:
                            dNoisy_prev = d + self.sigmas[t_prev] * torch.randn_like(d)

                        input_cat_prev = torch.cat((cond, dNoisy_prev), dim=1)
                        predicted_prev = self._apply_unet(
                            input_cat_prev, t_prev, step_idx=i + 1
                        )[:, cond.shape[1]:]

                        if i + 1 == self.nTimesteps - 1:
                            # At max t the model outputs x0 directly
                            x0_tmp = predicted_prev
                        else:
                            x0_tmp = dNoisy_prev - self.sigmas[t_prev] * predicted_prev

                        # Re-noise the lookahead x0 estimate to the current level
                        dNoisy = x0_tmp + self.sigmas[t] * torch.randn_like(d)

                    elif input_type == "own-pred":
                        # Forward-noise d, then iterate own-pred own_pred_iters times
                        dNoisy = d + self.sigmas[t] * torch.randn_like(d)
                        for _ in range(own_pred_iters):
                            input_cat_tmp = torch.cat((cond, dNoisy), dim=1)
                            predicted_tmp = self._apply_unet(
                                input_cat_tmp, t, step_idx=i
                            )[:, cond.shape[1]:]
                            x0_tmp = dNoisy - self.sigmas[t] * predicted_tmp
                            dNoise = torch.randn_like(d)
                            dNoisy = x0_tmp + self.sigmas[t] * dNoise

                    else:  # "ancestor" (default)
                        # Re-noise the previous x0 estimate at noise level t
                        dNoise = torch.randn_like(d)
                        dNoisy = x0_estimate + self.sigmas[t] * dNoise

                    if return_noise_pred:
                        targets.append(dNoise)

                # --- Standard refinement step ---
                input_cat = torch.cat((cond, dNoisy), dim=1)
                predicted = self._apply_unet(input_cat, t, step_idx=i)[:, cond.shape[1]:]

                if return_noise_pred:
                    predictions.append(predicted)

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
            if return_noise_pred:
                return x0_estimate, targets, predictions
            return x0_estimate
