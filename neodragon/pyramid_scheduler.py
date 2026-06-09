# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class SchedulerOutput(BaseOutput):
    # BaseOutput provides easy-dict like functionality
    prev_sample: torch.FloatTensor


class PyramidFlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        stages: int = 3,
        gamma: float = 1 / 3,
        stage_range: Optional[List[float]] = None,
    ) -> None:

        if stage_range is None:
            # Uniformly divided sigma values for the stage ranges
            stage_range = [i / stages for i in range(stages + 1)]
            self.register_to_config(stage_range=stage_range)

        # The timestep ratio for each stage
        self.timestep_ratios = {}
        # The detailed timesteps per stage
        self.timesteps_per_stage = {}
        # For caching the t_max - t_min values per stage
        self.timestep_per_stage_range = {}

        # Compute all the sigmas for different stages:
        self.end_sigmas = {}
        # these are based on the stage_range
        self.orig_start_sigmas = {}
        self.start_sigmas = {}  # NOTE: these are computed using the correction:
        # The beautiful equation: sk = 2ek / (1 + ek)
        # Check the derivation leading to equation 26 in the paper https://arxiv.org/pdf/2410.05954
        # NOTE: REMEMBER, the sigma range's notation is inverted in the paper for some reason :D
        self.sigmas_per_stage = {}
        self._init_sigmas_for_each_stage()

    @property
    def gamma(self) -> float:
        return self.config.gamma

    def _init_sigmas(self) -> None:
        num_train_timesteps = self.config.num_train_timesteps

        timesteps = np.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        # sigmas and timesteps
        self.sigmas = timesteps / num_train_timesteps
        # making sure there are no round-off errors
        self.timesteps = self.sigmas * num_train_timesteps

    def _init_sigmas_for_each_stage(self) -> None:
        self._init_sigmas()

        stage_distance = []
        stages = self.config.stages
        training_steps = self.config.num_train_timesteps
        stage_range = self.config.stage_range

        # Init the start and end point of each stage
        for i_s in range(stages):
            # To decide the start and ends point
            start_indice = int(stage_range[i_s] * training_steps)
            start_indice = max(start_indice, 0)
            end_indice = int(stage_range[i_s + 1] * training_steps)
            end_indice = min(end_indice, training_steps)
            start_sigma = self.sigmas[start_indice].item()
            end_sigma = (
                self.sigmas[end_indice].item() if end_indice < training_steps else 0.0
            )
            self.orig_start_sigmas[i_s] = start_sigma

            if i_s != 0:
                ori_sigma = 1 - start_sigma
                gamma = self.config.gamma
                corrected_sigma = (
                    1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                ) * ori_sigma
                start_sigma = 1 - corrected_sigma

            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas[i_s] = start_sigma
            self.end_sigmas[i_s] = end_sigma

        # Determine the ratio of each stage according to flow length
        tot_distance = sum(stage_distance)
        for i_s in range(stages):
            if i_s == 0:
                start_ratio = 0.0
            else:
                start_ratio = sum(stage_distance[:i_s]) / tot_distance
            if i_s == stages - 1:
                end_ratio = 1.0
            else:
                end_ratio = sum(stage_distance[: i_s + 1]) / tot_distance

            self.timestep_ratios[i_s] = (start_ratio, end_ratio)

        # Determine the timesteps and sigmas for each stage
        for i_s in range(stages):
            timestep_ratio = self.timestep_ratios[i_s]
            timestep_max = self.timesteps[int(timestep_ratio[0] * training_steps)]
            timestep_min = self.timesteps[
                min(int(timestep_ratio[1] * training_steps), training_steps - 1)
            ]
            timesteps = np.linspace(
                timestep_max,
                timestep_min,
                training_steps + 1,
            )
            self.timesteps_per_stage[i_s] = (
                timesteps[:-1]
                if isinstance(timesteps, torch.Tensor)
                else torch.from_numpy(timesteps[:-1])
            )
            self.timestep_per_stage_range[i_s] = (
                self.timesteps_per_stage[i_s][0] - self.timesteps_per_stage[i_s][-1]
            )

            # Local sigmas are always between (1 and 0)s uniformly sampled:
            stage_sigmas = np.linspace(
                1,
                0,
                training_steps + 1,
            )

            self.sigmas_per_stage[i_s] = torch.from_numpy(stage_sigmas[:-1])

    def get_stage_timesteps(
        self,
        num_inference_steps: int,
        stage: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.FloatTensor:
        stage_timesteps = self.timesteps_per_stage[stage]
        timestep_max = stage_timesteps[0].item()
        timestep_min = stage_timesteps[-1].item()

        timesteps = np.linspace(
            timestep_max,
            timestep_min,
            num_inference_steps,
        )
        timesteps = torch.from_numpy(timesteps).to(device=device)

        return timesteps

    def get_stage_sigmas(
        self,
        num_inference_steps: int,
        stage: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.FloatTensor:
        stage_sigmas = self.sigmas_per_stage[stage]
        sigma_max = stage_sigmas[0].item()
        sigma_min = stage_sigmas[-1].item()

        ratios = np.linspace(sigma_max, sigma_min, num_inference_steps)
        sigmas = torch.from_numpy(ratios).to(device=device)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        return sigmas

    def step(
        self,
        model_output: torch.FloatTensor,
        sigma: Union[float, torch.FloatTensor],
        sigma_next: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        prev_sample = sample + (sigma_next - sigma) * model_output
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
