# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Tuple

import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config


def _get_activation_fn(activation: str) -> nn.Module:
    if activation.lower() == "silu":
        return nn.SiLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


def _init_weights_glorot_uniform(m: nn.Module) -> None:
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0.0)


class SkipMLP(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        input_dims: int = 4096,
        output_dims: int = 1536,
        layer_depths: Tuple[int, ...] = (4096, 4096, 4096, 4096),
        skips: Tuple[bool, ...] = (True, True, True, True),
        use_biases: bool = True,
        pre_activation: bool = False,
        activation_fn: str = "gelu",
        out_activation_fn: str = "linear",
    ) -> None:
        super().__init__()

        self.network_layers = self._get_sequential_layers_with_skips()
        self.apply(_init_weights_glorot_uniform)

    def _get_sequential_layers_with_skips(self) -> nn.ModuleList:
        modules, in_features = [], self.config.input_dims

        assert len(self.config.layer_depths) == len(
            self.config.skips
        ), "The number of layer depths and skips must match."

        # Add the requested layers
        for skip, layer_depth in zip(self.config.skips, self.config.layer_depths):
            modules.append(
                nn.Sequential(
                    (
                        _get_activation_fn(self.config.activation_fn)
                        if self.config.pre_activation
                        else _get_activation_fn("linear")
                    ),
                    nn.Linear(
                        in_features=in_features,
                        out_features=layer_depth,
                        bias=self.config.use_biases,
                    ),
                    (
                        _get_activation_fn("linear")
                        if self.config.pre_activation
                        else _get_activation_fn(self.config.activation_fn)
                    ),
                )
            )
            in_features = layer_depth + self.config.input_dims if skip else layer_depth

        # Add the final output layer separately
        modules.append(
            nn.Sequential(
                nn.Linear(
                    in_features=in_features,
                    out_features=self.config.output_dims,
                    bias=self.config.use_biases,
                ),
                _get_activation_fn(self.config.out_activation_fn),
            )
        )

        return nn.ModuleList(modules)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        y = self.network_layers[0](x)  # First layer doesn't have skips

        # Apply the rest of the layers with skips
        for skip, network_layer in zip(self.config.skips, self.network_layers[1:]):
            y = torch.cat([x, y], dim=-1) if skip else y
            y = network_layer(y)

        return y


class ContextAdapter(SkipMLP):
    # ContextAdapter is nothing but a SkipMLP
    pass
