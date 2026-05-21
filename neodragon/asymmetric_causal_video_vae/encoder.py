from typing import Tuple

import torch
import torch.nn as nn

from .modeling_block import CausalUNetMidBlock2D, get_down_block
from .modeling_causal_ops import CausalConv3d, CausalGroupNorm


class CausalVaeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlockCausal3D",),
        spatial_down_sample: Tuple[bool, ...] = (True,),
        temporal_down_sample: Tuple[bool, ...] = (False,),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: Tuple[int, ...] = (2,),
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        block_dropout: Tuple[int, ...] = (0.0,),
        mid_block_add_attention=True,
    ) -> None:

        self._validate_config(
            down_block_types,
            spatial_down_sample,
            temporal_down_sample,
            block_out_channels,
            layers_per_block,
            block_dropout,
        )

        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                add_spatial_downsample=spatial_down_sample[i],
                add_temporal_downsample=temporal_down_sample[i],
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                dropout=block_dropout[i],
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = CausalUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            dropout=block_dropout[-1],
        )

        # out
        self.conv_norm_out = CausalGroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(
            block_out_channels[-1], conv_out_channels, kernel_size=3, stride=1
        )

    @staticmethod
    def _validate_config(
        down_block_types: Tuple[str, ...],
        spatial_down_sample: Tuple[bool, ...],
        temporal_down_sample: Tuple[bool, ...],
        block_out_channels: Tuple[int, ...],
        layers_per_block: Tuple[int, ...],
        block_dropout: Tuple[float, ...],
    ) -> None:
        num_blocks = len(down_block_types)
        assert (
            len(spatial_down_sample) == num_blocks
        ), f"spatial_down_sample length {len(spatial_down_sample)} != down_block_types length {num_blocks}"
        assert (
            len(temporal_down_sample) == num_blocks
        ), f"temporal_down_sample length {len(temporal_down_sample)} != down_block_types length {num_blocks}"
        assert (
            len(block_out_channels) == num_blocks
        ), f"block_out_channels length {len(block_out_channels)} != down_block_types length {num_blocks}"
        assert (
            len(layers_per_block) == num_blocks
        ), f"layers_per_block length {len(layers_per_block)} != down_block_types length {num_blocks}"
        assert (
            len(block_dropout) == num_blocks
        ), f"block_dropout length {len(block_dropout)} != down_block_types length {num_blocks}"

    def forward(
        self, x: torch.FloatTensor, is_init_image=True, temporal_chunk=False
    ) -> torch.FloatTensor:
        # conv-in preprocess
        x = self.conv_in(x, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        # down
        for down_block in self.down_blocks:
            x = down_block(x, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        # middle
        x = self.mid_block(x, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        # conv-out postprocess
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        return x
