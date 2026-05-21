# Copyright (c) 2026 Qualcomm Technologies, Inc.
# All Rights Reserved.

from pathlib import Path
from typing import Any, Dict, Optional

from diffusers import (
    AutoencoderKL,
    LCMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

SSD1B_VAE_ID = "ssd_1b_vae"
SSD1B_UNET_ID = "ssd_1b_unet"
SSD1B_TEXT_ENCODER_ID = "ssd_1b_text_encoder"
SSD1B_TEXT_ENCODER_2_ID = "ssd_1b_text_encoder_2"
SSD1B_TOKENIZER_ID = "ssd_1b_tokenizer"
SSD1B_TOKENIZER_2_ID = "ssd_1b_tokenizer_2"

DEFAULT_INFERENCE_PARAMS = {
    "num_inference_steps": 4,
    "save_intermediate": False,
    "timesteps": [999, 749, 499, 249],
    "height": 640,
    "width": 1024,
    "guidance_scale": 0.0,
}


class SSD1B_FirstFrameGeneratorPipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        default_inference_params: Dict = DEFAULT_INFERENCE_PARAMS,
    ) -> None:

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )

        self.default_inference_params = default_inference_params

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "SSD1B_FirstFrameGeneratorPipeline":
        # Load the rest of the components:
        vae = AutoencoderKL.from_pretrained(
            Path(pretrained_model_name_or_path) / SSD1B_VAE_ID, **kwargs
        )
        unet = UNet2DConditionModel.from_pretrained(
            Path(pretrained_model_name_or_path) / SSD1B_UNET_ID, **kwargs
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            Path(pretrained_model_name_or_path) / SSD1B_TOKENIZER_ID, **kwargs
        )
        text_encoder = CLIPTextModel.from_pretrained(
            Path(pretrained_model_name_or_path) / SSD1B_TEXT_ENCODER_ID, **kwargs
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            Path(pretrained_model_name_or_path) / SSD1B_TOKENIZER_2_ID, **kwargs
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            Path(pretrained_model_name_or_path) / SSD1B_TEXT_ENCODER_2_ID, **kwargs
        )
        scheduler = LCMScheduler(
            set_alpha_to_one=False,
            original_inference_steps=len(DEFAULT_INFERENCE_PARAMS["timesteps"]),
            steps_offset=1,
        )

        return cls(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )

    def __call__(self, *args, **kwargs):
        merged_kwargs = {**self.default_inference_params, **kwargs}
        return super().__call__(*args, **merged_kwargs)
