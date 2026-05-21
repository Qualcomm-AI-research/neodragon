import warnings
from typing import Any, List, Optional, Tuple

import torch
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from einops import rearrange
from torch import nn

from .modeling_normalization import AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(
        self, hidden_states: torch.FloatTensor, *args: Any, **kwargs: Any
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored. "
                "Please remove it, as passing it will raise an error in the future. "
                "`scale` should directly be passed while calling the underlying pipeline component "
                "i.e., via `cross_attention_kwargs`."
            )
            warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class VarlenSelfAttentionWithT5Mask:
    def apply_rope(
        self, xq: torch.FloatTensor, xk: torch.FloatTensor, freqs_cis: torch.FloatTensor
    ) -> torch.FloatTensor:
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def __call__(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        encoder_query: torch.FloatTensor,
        encoder_key: torch.FloatTensor,
        encoder_value: torch.FloatTensor,
        hidden_length: Optional[List[int]] = None,
        image_rotary_emb: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        assert attention_mask is not None, "The attention mask needed to be set"

        encoder_length = encoder_query.shape[1]
        num_stages = len(hidden_length)

        encoder_qkv = torch.stack(
            [encoder_query, encoder_key, encoder_value], dim=2
        )  # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2)  # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        output_encoder_hidden_list = []
        output_hidden_list = []

        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum : i_sum + length]
            concat_qkv_tokens = torch.cat(
                [encoder_qkv_tokens, qkv_tokens], dim=1
            )  # [bs, tot_seq, 3, nhead, dim]

            if image_rotary_emb is not None:
                concat_qkv_tokens[:, :, 0], concat_qkv_tokens[:, :, 1] = self.apply_rope(
                    concat_qkv_tokens[:, :, 0],
                    concat_qkv_tokens[:, :, 1],
                    image_rotary_emb[i_p],
                )

            query, key, value = concat_qkv_tokens.unbind(2)  # [bs, tot_seq, nhead, dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
            stage_hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2).flatten(
                2, 3
            )  # [bs, tot_seq, dim]

            output_encoder_hidden_list.append(stage_hidden_states[:, :encoder_length])
            output_hidden_list.append(stage_hidden_states[:, encoder_length:])
            i_sum += length

        output_encoder_hidden = torch.stack(output_encoder_hidden_list, dim=1)  # [b n s d]
        output_encoder_hidden = rearrange(output_encoder_hidden, "b n s d -> (b n) s d")
        output_hidden = torch.cat(output_hidden_list, dim=1)

        return output_hidden, output_encoder_hidden


class JointAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: bool = False,
    ) -> None:
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.use_bias = bias
        self.dropout = dropout

        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only

        self.scale = dim_head**-0.5
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)

            if qk_norm is None:
                self.norm_add_q = None
                self.norm_add_k = None
            elif qk_norm == "layer_norm":
                self.norm_add_q = nn.LayerNorm(dim_head, eps=eps)
                self.norm_add_k = nn.LayerNorm(dim_head, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_add_q = RMSNorm(dim_head, eps=eps)
                self.norm_add_k = RMSNorm(dim_head, eps=eps)
            else:
                raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        self.var_len_attn = VarlenSelfAttentionWithT5Mask()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,  # [B, L, S]
        hidden_length: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        **kwargs: Any,  # for compatible API
    ) -> torch.FloatTensor:
        # This function is only used during training
        # `sample` projections.
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(query.shape[0], -1, self.heads, head_dim)
        key = key.view(key.shape[0], -1, self.heads, head_dim)
        value = value.view(value.shape[0], -1, self.heads, head_dim)

        if self.norm_q is not None:
            query = self.norm_q(query)

        if self.norm_k is not None:
            key = self.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            encoder_hidden_states_query_proj.shape[0], -1, self.heads, head_dim
        )
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            encoder_hidden_states_key_proj.shape[0], -1, self.heads, head_dim
        )
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            encoder_hidden_states_value_proj.shape[0], -1, self.heads, head_dim
        )

        if self.norm_add_q is not None:
            encoder_hidden_states_query_proj = self.norm_add_q(encoder_hidden_states_query_proj)

        if self.norm_add_k is not None:
            encoder_hidden_states_key_proj = self.norm_add_k(encoder_hidden_states_key_proj)

        hidden_states, encoder_hidden_states = self.var_len_attn(
            query,
            key,
            value,
            encoder_hidden_states_query_proj,
            encoder_hidden_states_key_proj,
            encoder_hidden_states_value_proj,
            hidden_length,
            image_rotary_emb,
            attention_mask,
        )

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        if not self.context_pre_only:
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class JointTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: Optional[str] = None,
        context_pre_only: bool = False,
    ) -> None:
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim,
                dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="layer_norm",
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        self.attn = JointAttention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim // num_attention_heads,
            heads=num_attention_heads,
            out_dim=attention_head_dim,
            qk_norm=qk_norm,
            context_pre_only=context_pre_only,
            bias=True,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        temb: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        hidden_length: Optional[List[int]] = None,
        image_rotary_emb: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb, hidden_length=hidden_length
        )

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(
                encoder_hidden_states,
                emb=temb,
            )

        # Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
            hidden_length=hidden_length,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = (
                norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = (
                encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            )

        return encoder_hidden_states, hidden_states
