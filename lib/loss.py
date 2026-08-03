#!/usr/bin/env python
# coding=utf-8
"""Flow-matching training objective for SD3 (as in diffusers' train_controlnet_sd3.py)."""

import copy

import torch

from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


class SD3FlowMatchLoss:
    """Timestep sampling, noising and loss weighting for SD3 flow matching.

    Bundles what the upstream example spreads across its training loop:
    density-weighted timestep sampling, per-timestep sigma lookup, the
    zt = (1 - t) * x + t * z1 noising, EDM-style output preconditioning
    (Section 5 of https://huggingface.co/papers/2206.00364) and the
    weighted MSE.
    """

    def __init__(
        self,
        noise_scheduler,
        weighting_scheme="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        precondition_outputs=True,
    ):
        # Private copy so sampling elsewhere can't mutate the schedule state
        # (upstream keeps a deepcopy for the same reason).
        self.scheduler = copy.deepcopy(noise_scheduler)
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale
        self.precondition_outputs = precondition_outputs

    def sample_timesteps(self, batch_size, device):
        """One density-weighted random timestep per sample (non-uniform for
        weighting schemes like logit_normal)."""
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        return self.scheduler.timesteps[indices].to(device=device)

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def add_noise(self, model_input, noise, sigmas):
        """Flow-matching forward process: zt = (1 - texp) * x + texp * z1."""
        return (1.0 - sigmas) * model_input + sigmas * noise

    def __call__(self, model_pred, model_input, noise, noisy_model_input, sigmas):
        if self.precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_model_input
            target = model_input
        else:
            target = noise - model_input

        # These weighting schemes use uniform timestep sampling and post-weight the loss.
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        return loss.mean()
