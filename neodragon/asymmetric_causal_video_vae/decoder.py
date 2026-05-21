"""
---------------------------------------------------------------------------------------
|                                !!! ORIGINAL LICENSE !!!                             |
---------------------------------------------------------------------------------------
|    MIT License                                                                      |
|                                                                                     |
|    Copyright (c) 2025 Ollin Boer Bohan                                              |
|                                                                                     |
|    Permission is hereby granted, free of charge, to any person obtaining a copy     |
|    of this software and associated documentation files (the "Software"), to deal    |
|    in the Software without restriction, including without limitation the rights     |
|    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell        |
|    copies of the Software, and to permit persons to whom the Software is            |
|    furnished to do so, subject to the following conditions:                         |
|                                                                                     |
|    The above copyright notice and this permission notice shall be included in all   |
|    copies or substantial portions of the Software.                                  |
|                                                                                     |
|    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR       |
|    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,         |
|    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE      |
|    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER           |
|    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,    |
|    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE    |
|    SOFTWARE.                                                                        |
---------------------------------------------------------------------------------------

This code has been adapted from https://github.com/madebyollin/taehv/blob/main/taehv.py
"""

from collections import namedtuple
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in: int, n_out: int, **kwargs) -> nn.Module:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.FloatTensor, past: torch.FloatTensor) -> torch.FloatTensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks(
    model: nn.Sequential, x: torch.FloatTensor, parallel: bool, show_progress_bar: bool
) -> torch.FloatTensor:
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N * T, C, H, W)
        # parallel over input timesteps, iterate over blocks
        for b in tqdm(model, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        # TODO(oboerbohan): at least on macos this still gradually uses more memory during decode...
        # need to fix :(
        out = []
        # iterate over input timesteps and also iterate over blocks.
        # because of the cursed TPool/TGrow blocks, this is not a nested loop,
        # it's actually a ***graph traversal*** problem! so let's make a queue
        work_queue = [
            TWorkItem(xt, 0) for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))
        ]
        # in addition to manually managing our queue, we also need to manually manage our progressbar.
        # we'll update it for every source node that we consume.
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        # we'll also need a separate addressable memory per node as well
        mem = [None] * len(model)
        while work_queue:
            xt, i = work_queue.pop(0)
            if i == 0:
                # new source node consumed
                progress_bar.update(1)
            if i == len(model):
                # reached end of the graph, append result to output list
                out.append(xt)
            else:
                # fetch the block to process
                b = model[i]
                if isinstance(b, MemBlock):
                    # mem blocks are simple since we're visiting the graph in causal order
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(
                            xt
                        )  # inplace might reduce mysterious pytorch memory allocations? doesn't help though
                    # add successor to work queue
                    work_queue.insert(0, TWorkItem(xt_new, i + 1))
                elif isinstance(b, TPool):
                    # pool blocks are miserable
                    if mem[i] is None:
                        mem[i] = []  # pool memory is itself a queue of inputs to pool
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        # pool mem is in invalid state, we should have pooled before this
                        raise ValueError("???")
                    if len(mem[i]) < b.stride:
                        # pool mem is not yet full, go back to processing the work queue
                        pass
                    else:
                        # pool mem is ready, run the pool block
                        N, C, H, W = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(N * b.stride, C, H, W))
                        # reset the pool mem
                        mem[i] = []
                        # add successor to work queue
                        work_queue.insert(0, TWorkItem(xt, i + 1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C, H, W = xt.shape
                    # each tgrow has multiple successor nodes
                    for xt_next in reversed(xt.view(N, b.stride * C, H, W).chunk(b.stride, 1)):
                        # add successor to work queue
                        work_queue.insert(0, TWorkItem(xt_next, i + 1))
                else:
                    # normal block with no funny business
                    xt = b(xt)
                    # add successor to work queue
                    work_queue.insert(0, TWorkItem(xt, i + 1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x


class TAEHVDecoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        latent_channels: int = 16,
        n_f: Tuple[int, ...] = (256, 128, 64, 64),
        decoder_time_upscale: Tuple[bool, ...] = (True, True, True),
        decoder_space_upscale: Tuple[bool, ...] = (True, True, True),
    ) -> None:
        super().__init__()

        self.latent_channels = latent_channels
        self.image_channels = image_channels
        self.n_f = n_f
        self.decoder_time_upscale = decoder_time_upscale
        self.decoder_space_upscale = decoder_space_upscale

        self.blocks = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1),
            conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            conv(n_f[3], self.image_channels),
        )
        self.frames_to_trim = 2 ** sum(self.decoder_time_upscale) - 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.transpose(1, 2)
        x = apply_model_with_memblocks(self.blocks, x, parallel=True, show_progress_bar=False)
        x = x[:, self.frames_to_trim :]
        x = x.transpose(1, 2)
        return x
