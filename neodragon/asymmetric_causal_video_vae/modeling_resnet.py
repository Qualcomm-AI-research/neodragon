from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.normalization import AdaGroupNorm

from .modeling_causal_ops import CausalConv3d, CausalGroupNorm


class CausalResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",  # default, scale_shift, ada_group, spatial
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels

        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = CausalGroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = CausalGroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = CausalConv3d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1)

        self.nonlinearity = get_activation(non_linearity)
        self.upsample = self.downsample = None
        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor = None,
        is_init_image=True,
        temporal_chunk=False,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(
            hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk
        )

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(
            hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk
        )

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(
                input_tensor, is_init_image=is_init_image, temporal_chunk=temporal_chunk
            )

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class ResnetBlock2D(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",  # default, scale_shift, ada_group, spatial
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels

        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        conv_cls = nn.Conv3d

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )

        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = conv_cls(
            out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1
        )

        self.nonlinearity = get_activation(non_linearity)
        self.upsample = self.downsample = None
        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class CausalDownsample2x(nn.Module):

    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size=3,
        bias=True,
    ):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        stride = (1, 2, 2)
        self.name = name

        if use_conv:
            conv = CausalConv3d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(
        self, hidden_states: torch.FloatTensor, is_init_image=True, temporal_chunk=False
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(
            hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk
        )
        return hidden_states


class Downsample2D(nn.Module):

    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
        padding: int = 0,
        name: str = "conv",
        kernel_size=3,
        bias=True,
    ):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding

        stride = (1, 2, 2)
        self.name = name
        conv_cls = nn.Conv3d

        if use_conv:
            conv = conv_cls(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1, 1, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class TemporalDownsample2x(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 0,
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding

        stride = (2, 1, 1)

        if use_conv:
            conv = nn.Conv3d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            raise NotImplementedError("Not implemented for temporal downsample without")

        self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv and self.padding == 0:
            if hidden_states.shape[2] == 1:
                # image
                pad = (1, 1, 1, 1, 1, 1)
            else:
                # video
                pad = (1, 1, 1, 1, 0, 1)

            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        hidden_states = self.conv(hidden_states)
        return hidden_states


class CausalTemporalDownsample2x(nn.Module):

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = (2, 1, 1)

        if use_conv:
            conv = CausalConv3d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            )
        else:
            raise NotImplementedError("Not implemented for temporal downsample without convolution")

        self.conv = conv

    def forward(
        self, hidden_states: torch.FloatTensor, is_init_image=True, temporal_chunk=False
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(
            hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk
        )
        return hidden_states
