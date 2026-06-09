# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, List, Tuple

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5Tokenizer

from .distil_t5 import T5EncoderWithProjection

MAX_SEQUENCE_LENGTH = 128
TOKENIZER_ID = "tokenizer"
TEXT_ENCODER_ID = "text_encoder"
TOKENIZER_2_ID = "tokenizer_2"
TEXT_ENCODER_2_ID = "text_encoder_2"
TOKENIZER_3_ID = "tokenizer_3"
TEXT_ENCODER_3_ID = "text_encoder_3"


class TextEncoderBundle(ModelMixin, ConfigMixin):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_3: T5Tokenizer,
        text_encoder_3: T5EncoderWithProjection,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_3 = tokenizer_3
        self.text_encoder_3 = text_encoder_3

        # shorthand for max length
        self.tokenizer_max_length = self.tokenizer.model_max_length
        self.max_sequence_length = MAX_SEQUENCE_LENGTH

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> "TextEncoderBundle":
        # load all components
        tokenizer = CLIPTokenizer.from_pretrained(
            f"{pretrained_model_name_or_path}/{TOKENIZER_ID}", **kwargs
        )
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            f"{pretrained_model_name_or_path}/{TEXT_ENCODER_ID}", **kwargs
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            f"{pretrained_model_name_or_path}/{TOKENIZER_2_ID}", **kwargs
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            f"{pretrained_model_name_or_path}/{TEXT_ENCODER_2_ID}", **kwargs
        )
        tokenizer_3 = T5Tokenizer.from_pretrained(
            f"{pretrained_model_name_or_path}/{TOKENIZER_3_ID}", **kwargs
        )
        text_encoder_3 = T5EncoderWithProjection.from_pretrained(
            f"{pretrained_model_name_or_path}/{TEXT_ENCODER_3_ID}", **kwargs
        )
        return cls(
            tokenizer,
            text_encoder,
            tokenizer_2,
            text_encoder_2,
            tokenizer_3,
            text_encoder_3,
        )

    def forward(
        self,
        input_prompts: List[str],
        device: torch.device,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        # obtain the clip_prompt embeddings:
        pooled_prompt_embeds = []
        for tokenizer, text_encoder in [
            (self.tokenizer, self.text_encoder),
            (self.tokenizer_2, self.text_encoder_2),
        ]:
            text_inputs = tokenizer(
                input_prompts,
                padding="max_length",
                max_length=self.tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            prompt_embed = text_encoder(
                text_input_ids.to(device), output_hidden_states=True
            )[0]
            pooled_prompt_embeds.append(prompt_embed)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=-1)

        # obtain the t5 prompt embeddings:
        text_inputs = self.tokenizer_3(
            input_prompts,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)
        prompt_embeds = self.text_encoder_3(
            text_input_ids.to(device), attention_mask=prompt_attention_mask
        )[0]
        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds
