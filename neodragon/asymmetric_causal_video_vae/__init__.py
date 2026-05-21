# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from timm.layers import trunc_normal_

from .decoder import TAEHVDecoder
from .encoder import CausalVaeEncoder
from .modeling_causal_ops import CausalConv3d


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def mode(self) -> torch.Tensor:
        return self.mean


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor


class AsymmetricCausalVideoVAE(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        # encoder related parameters
        encoder_in_channels: int = 3,
        encoder_out_channels: int = 16,
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        encoder_down_block_types: Tuple[str, ...] = (
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
        ),
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        encoder_spatial_down_sample: Tuple[bool, ...] = (True, True, True, False),
        encoder_temporal_down_sample: Tuple[bool, ...] = (True, True, True, False),
        encoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        encoder_act_fn: str = "silu",
        encoder_norm_num_groups: int = 32,
        # decoder related  (Disabled atm)
        decoder_num_features: Tuple[int, ...] = (256, 128, 64, 64),
        decoder_time_upscale: Tuple[bool, ...] = (True, True, True),
        decoder_space_upscale: Tuple[bool, ...] = (True, True, True),
        # tiling related
        sample_size: int = 256,  # resolution at which model is trained
        temporal_downsample_scale: int = 8,
        spatial_downsample_scale: int = 8,
    ):
        super().__init__()

        print(f"The latent dimmension channels are {encoder_out_channels}")

        # build the encoder
        self.encoder = CausalVaeEncoder(
            in_channels=encoder_in_channels,
            out_channels=encoder_out_channels,
            down_block_types=encoder_down_block_types,
            spatial_down_sample=encoder_spatial_down_sample,
            temporal_down_sample=encoder_temporal_down_sample,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            act_fn=encoder_act_fn,
            norm_num_groups=encoder_norm_num_groups,
            double_z=True,  # is a must
            block_dropout=encoder_block_dropout,
        )

        # build the decoder:
        self.decoder = TAEHVDecoder(
            image_channels=encoder_in_channels,
            latent_channels=encoder_out_channels,
            n_f=decoder_num_features,
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
        )

        # Quant convolutions:
        self.quant_conv = CausalConv3d(
            2 * encoder_out_channels, 2 * encoder_out_channels, kernel_size=1, stride=1
        )
        self.post_quant_conv = CausalConv3d(
            encoder_out_channels, encoder_out_channels, kernel_size=1, stride=1
        )

        # tiling is disabled by default (But can be enabled externally)
        self.use_tiling = False
        self._setup_tiling(temporal_downsample_scale, spatial_downsample_scale)

        self.apply(self._init_weights)

    def _setup_tiling(self, downsample_scale: int, spatial_downsample_scale: int) -> None:
        self.tile_sample_min_size = self.config.sample_size

        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / downsample_scale)
        self.encode_tile_overlap_factor = 1 / 4
        self.decode_tile_overlap_factor = 1 / 4
        self.downsample_scale = downsample_scale
        self.spatial_downsample_scale = spatial_downsample_scale

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
                :, :, :, y, :
            ] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                :, :, :, :, x
            ] * (x / blend_extent)
        return b

    def tiled_encode(
        self,
        x: torch.FloatTensor,
        return_dict: bool = True,
        temporal_chunk=False,
        window_size=16,
    ) -> AutoencoderKLOutput:
        overlap_size = int(self.tile_sample_min_size * (1 - self.encode_tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.encode_tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                if temporal_chunk:
                    tile = self.temporal_chunk_encode(tile, window_size=window_size)
                else:
                    tile = self.encoder(tile, is_init_image=True, temporal_chunk=False)
                    tile = self.quant_conv(tile, is_init_image=True, temporal_chunk=False)
                row.append(tile)
            rows.append(row)

        # Blend the individual tiles back together
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @torch.no_grad()
    def temporal_chunk_encode(self, x: torch.FloatTensor, window_size=16):
        num_frames = x.shape[2]
        init_window_size = window_size + 1
        frame_list = [x[:, :, :init_window_size]]

        # To chunk the long video
        full_chunk_size = (num_frames - init_window_size) // window_size
        fid = init_window_size
        for idx in range(full_chunk_size):
            frame_list.append(x[:, :, fid : fid + window_size])
            fid += window_size

        if fid < num_frames:
            frame_list.append(x[:, :, fid:])

        latent_list = []
        for idx, frames in enumerate(frame_list):
            if idx == 0:
                h = self.encoder(frames, is_init_image=True, temporal_chunk=True)
                moments = self.quant_conv(h, is_init_image=True, temporal_chunk=True)
            else:
                h = self.encoder(frames, is_init_image=False, temporal_chunk=True)
                moments = self.quant_conv(h, is_init_image=False, temporal_chunk=True)

            latent_list.append(moments)

        latent = torch.cat(latent_list, dim=2)
        return latent

    def encode(
        self,
        x: torch.FloatTensor,
        return_dict: bool = True,
        is_init_image=True,
        temporal_chunk=False,
        window_size=16,
        tile_sample_min_size=256,
    ) -> AutoencoderKLOutput:
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)

        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
        ):
            return self.tiled_encode(
                x,
                return_dict=return_dict,
                temporal_chunk=temporal_chunk,
                window_size=window_size,
            )

        if temporal_chunk:
            moments = self.temporal_chunk_encode(x, window_size=window_size)
        else:
            h = self.encoder(x, is_init_image=is_init_image, temporal_chunk=False)
            moments = self.quant_conv(h, is_init_image=is_init_image, temporal_chunk=False)

        posterior = DiagonalGaussianDistribution(moments)

        if return_dict:
            return AutoencoderKLOutput(latent_dist=posterior)

        return (posterior,)

    def decode(
        self,
        z: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
        freeze_encoder: bool = False,
        is_init_image=True,
        temporal_chunk=False,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        x = sample

        if freeze_encoder:
            with torch.no_grad():
                posterior = self.encode(
                    x, is_init_image=is_init_image, temporal_chunk=temporal_chunk
                ).latent_dist
        else:
            posterior = self.encode(
                x, is_init_image=is_init_image, temporal_chunk=temporal_chunk
            ).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z).sample

        return posterior, dec
