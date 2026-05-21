import numbers
from typing import Optional, Tuple

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, elementwise_affine: bool = True) -> None:
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight

        hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward_with_pad(
        self,
        x: torch.FloatTensor,
        conditioning_embedding: torch.FloatTensor,
        hidden_length: Optional[Tuple[int, ...]] = None,
    ) -> torch.FloatTensor:
        assert hidden_length is not None

        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        batch_emb = torch.zeros_like(x).repeat(1, 1, 2)

        i_sum = 0
        num_stages = len(hidden_length)
        for i_p, length in enumerate(hidden_length):
            batch_emb[:, i_sum : i_sum + length] = emb[i_p::num_stages][:, None]
            i_sum += length

        batch_scale, batch_shift = torch.chunk(batch_emb, 2, dim=2)
        x = self.norm(x) * (1 + batch_scale) + batch_shift
        return x

    def forward(
        self,
        x: torch.FloatTensor,
        conditioning_embedding: torch.FloatTensor,
        hidden_length: Optional[Tuple[int, ...]] = None,
    ) -> torch.FloatTensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        if hidden_length is not None:
            return self.forward_with_pad(x, conditioning_embedding, hidden_length)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.emb = None
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward_with_pad(
        self,
        x: torch.FloatTensor,
        timestep: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.FloatTensor] = None,
        hidden_length: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        # x: [bs, seq_len, dim]
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)

        emb = self.linear(self.silu(emb))
        batch_emb = torch.zeros_like(x).repeat(1, 1, 6)

        i_sum = 0
        num_stages = len(hidden_length)
        for i_p, length in enumerate(hidden_length):
            batch_emb[:, i_sum : i_sum + length] = emb[i_p::num_stages][:, None]
            i_sum += length

        (
            batch_shift_msa,
            batch_scale_msa,
            batch_gate_msa,
            batch_shift_mlp,
            batch_scale_mlp,
            batch_gate_mlp,
        ) = batch_emb.chunk(6, dim=2)
        x = self.norm(x) * (1 + batch_scale_msa) + batch_shift_msa
        return x, batch_gate_msa, batch_shift_mlp, batch_scale_mlp, batch_gate_mlp

    def forward(
        self,
        x: torch.FloatTensor,
        timestep: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.FloatTensor] = None,
        hidden_length: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        if hidden_length is not None:
            return self.forward_with_pad(
                x, timestep, class_labels, hidden_dtype, emb, hidden_length
            )
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
