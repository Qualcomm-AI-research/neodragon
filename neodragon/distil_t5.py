"""
---------------------------------------------------------------------------------------
|                                !!! ORIGINAL LICENSE !!!                             |
---------------------------------------------------------------------------------------
|    MIT License                                                                      |
|                                                                                     |
|    Copyright © 2025 JD.com                                                          |
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

This code has been adapted from https://github.com/LifuWang-66/DistillT5/blob/main/models/T5_encoder.py
"""

from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from transformers import T5Config, T5EncoderModel, T5PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


class T5ProjectionConfig(T5Config):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.project_in_dim = kwargs.get("project_in_dim", 768)
        self.project_out_dim = kwargs.get("out_dim", 4096)


class T5EncoderWithProjection(T5PreTrainedModel):
    config_class = T5ProjectionConfig

    def __init__(self, config: T5ProjectionConfig) -> None:
        super().__init__(config)
        # self.encoder = encoder
        self.encoder = T5EncoderModel(config)

        self.final_projection = nn.Sequential(
            nn.Linear(config.project_in_dim, config.project_out_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.project_out_dim, config.project_out_dim, bias=False),
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # for peft support
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else False

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = self.final_projection(encoder_outputs[0])
        # last_hidden_state = self.final_block(last_hidden_state)[0]

        if not return_dict:
            return tuple(v for v in [last_hidden_state] if v is not None)

        return BaseModelOutput(last_hidden_state=last_hidden_state)
