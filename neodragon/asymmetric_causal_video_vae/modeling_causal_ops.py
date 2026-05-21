from collections import deque
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from torch import Tensor


def divisible_by(num, den):
    return (num % den) == 0


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def is_odd(n):
    return not divisible_by(n, 2)


class CausalGroupNorm(nn.GroupNorm):

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


class CausalConv3d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop("dilation", 1)
        self.pad_mode = pad_mode

        if isinstance(stride, int):
            stride = (stride, 1, 1)

        time_pad = dilation * (time_kernel_size - 1)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.temporal_stride = stride[0]
        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )
        self.time_uncausal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            0,
            0,
        )

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            **kwargs,
        )
        self.cache_front_feat = deque()

    def _clear_context_cache(self):
        del self.cache_front_feat
        self.cache_front_feat = deque()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, is_init_image=True, temporal_chunk=False):

        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        if not temporal_chunk:
            x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        else:
            assert not self.training, "The feature cache should not be used in training"
            if is_init_image:
                # Encode the first chunk
                x = F.pad(x, self.time_causal_padding, mode=pad_mode)
                self._clear_context_cache()
                self.cache_front_feat.append(x[:, :, -2:].clone().detach())
            else:
                # Encoder subsequent chunks
                x = F.pad(x, self.time_uncausal_padding, mode=pad_mode)
                video_front_context = self.cache_front_feat.pop()
                self._clear_context_cache()

                if self.temporal_stride == 1 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context, x], dim=2)
                elif self.temporal_stride == 2 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context[:, :, -1:], x], dim=2)

                self.cache_front_feat.append(x[:, :, -2:].clone().detach())

        x = self.conv(x)
        return x
