# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import torch

from einops import rearrange
from torch import nn
from typing import Optional

from diffusers.models.attention_processor import Attention
from diffusers.utils import logging

from .modeling_resnet import (
    CausalDownsample2x,
    CausalResnetBlock3D,
    CausalTemporalDownsample2x,
    Downsample2D,
    ResnetBlock2D,
    TemporalDownsample2x,
)

logger = logging.get_logger(__name__)


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int = None,
    add_spatial_downsample: bool = None,
    add_temporal_downsample: bool = None,
    resnet_eps: float = 1e-6,
    resnet_act_fn: str = "silu",
    resnet_groups: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    resnet_time_scale_shift: str = "default",
    dropout: float = 0.0,
) -> nn.Module:
    if down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_spatial_downsample=add_spatial_downsample,
            add_temporal_downsample=add_temporal_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    if down_block_type == "DownEncoderBlockCausal3D":
        return DownEncoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_spatial_downsample=add_spatial_downsample,
            add_temporal_downsample=add_temporal_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    raise ValueError(f"{down_block_type} does not exist.")


class CausalUNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = (
                resnet_groups if resnet_time_scale_shift == "default" else None
            )

        # there is always at least one resnet
        resnets = [
            CausalResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                # Spatial attention
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=(
                            temb_channels
                            if resnet_time_scale_shift == "spatial"
                            else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                CausalResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        is_init_image=True,
        temporal_chunk=False,
    ) -> torch.FloatTensor:
        hidden_states = self.resnets[0](
            hidden_states,
            temb,
            is_init_image=is_init_image,
            temporal_chunk=temporal_chunk,
        )
        t = hidden_states.shape[2]

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, "b c t h w -> b t c h w")
                hidden_states = rearrange(hidden_states, "b t c h w -> (b t) c h w")
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(
                    hidden_states, "(b t) c h w -> b t c h w", t=t
                )
                hidden_states = rearrange(hidden_states, "b t c h w -> b c t h w")

            hidden_states = resnet(
                hidden_states,
                temb,
                is_init_image=is_init_image,
                temporal_chunk=temporal_chunk,
            )

        return hidden_states


class DownEncoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_spatial_downsample: bool = True,
        add_temporal_downsample: bool = False,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                CausalResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_spatial_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    CausalDownsample2x(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                    )
                ]
            )
        else:
            self.downsamplers = None

        if add_temporal_downsample:
            self.temporal_downsamplers = nn.ModuleList(
                [
                    CausalTemporalDownsample2x(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                    )
                ]
            )
        else:
            self.temporal_downsamplers = None

    def forward(
        self, hidden_states: torch.FloatTensor, is_init_image=True, temporal_chunk=False
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states,
                temb=None,
                is_init_image=is_init_image,
                temporal_chunk=temporal_chunk,
            )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(
                    hidden_states,
                    is_init_image=is_init_image,
                    temporal_chunk=temporal_chunk,
                )

        if self.temporal_downsamplers is not None:
            for temporal_downsampler in self.temporal_downsamplers:
                hidden_states = temporal_downsampler(
                    hidden_states,
                    is_init_image=is_init_image,
                    temporal_chunk=temporal_chunk,
                )

        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_spatial_downsample: bool = True,
        add_temporal_downsample: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_spatial_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        if add_temporal_downsample:
            self.temporal_downsamplers = nn.ModuleList(
                [
                    TemporalDownsample2x(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                    )
                ]
            )
        else:
            self.temporal_downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        if self.temporal_downsamplers is not None:
            for temporal_downsampler in self.temporal_downsamplers:
                hidden_states = temporal_downsampler(hidden_states)

        return hidden_states
