# Copyright (c) 2026 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import register_to_config
from huggingface_hub import snapshot_download
from PIL.Image import Image

from .asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
from .context_adapter import ContextAdapter
from .first_frame_gen import SSD1B_FirstFrameGeneratorPipeline
from .pyramid_mmdit import PyramidMMDiT
from .pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
from .text_encoder_bundle import TextEncoderBundle
from .utils.generation_utils import generate, generate_hybrid

MULTISTEP_CONTEXT_ADAPTER_ID = "context_adapter_multistep_t2v"
CONTEXT_ADAPTER_ID = "context_adapter"
MULTISTEP_DIT_ID = "diffusion_transformer_320p_multistep_t2v"
DIT_ID = "diffusion_transformer_320p"
VAE_ID = "causal_video_vae"
PIPELINE_MODES = ["hybrid", "monolithic"]


COMMON_GENERATION_CONFIG = {
    "height": 320,
    "width": 512,
    "num_frames": 49,
}
HYBRID_GENERATION_CONFIG = {
    "num_inference_steps": [1, 1, 1],
    "video_num_inference_steps": [1, 1, 1],
    "do_classifier_free_guidance": False,
    "guidance_scale": 0.0,
    "video_guidance_scale": 0.0,
}
MONOLITHIC_GENERATION_CONFIG = {
    "num_inference_steps": [20, 20, 20],
    "video_num_inference_steps": [10, 10, 10],
    "do_classifier_free_guidance": True,
    "guidance_scale": 7.0,
    "video_guidance_scale": 5.0,
}


def _validate_mode(mode: str) -> None:
    if mode not in PIPELINE_MODES:
        raise ValueError(f"Invalid mode: {mode}. Supported modes are: {PIPELINE_MODES}")


class NeodragonPipeline(DiffusionPipeline):
    @register_to_config
    def __init__(
        self,
        # Main components:
        text_encoder_bundle: TextEncoderBundle,
        context_adapter: ContextAdapter,
        dit: PyramidMMDiT,
        vae: AsymmetricCausalVideoVAE,
        scheduler: PyramidFlowMatchEulerDiscreteScheduler,
        model_path: str = None,
        # Behavioral config:
        mode: str = "hybrid",
        # Pyramidal-Causal config:
        frames_per_unit: int = 1,
        stages: Tuple[int, ...] = (1, 2, 4),
        # Generation configurations:
        gen_confs: Dict[str, Any] = {
            "common": COMMON_GENERATION_CONFIG,
            "hybrid": HYBRID_GENERATION_CONFIG,
            "monolithic": MONOLITHIC_GENERATION_CONFIG,
        },
    ) -> None:
        _validate_mode(mode)
        super().__init__()

        self.model_path = model_path
        self.mode = mode

        # Register every component so Diffusers can save/load them
        self.register_modules(
            text_encoder_bundle=text_encoder_bundle,
            context_adapter=context_adapter,
            dit=dit,
            vae=vae,
            scheduler=scheduler,
        )

        self.first_frame_gen_pipeline = None
        if self.mode == "hybrid":
            self.first_frame_gen_pipeline = (
                SSD1B_FirstFrameGeneratorPipeline.from_pretrained(model_path)
                .to(self.dtype)
                .to(self.device)
            )

    def to(self, device: Union[str, torch.device], **kwargs) -> "NeodragonPipeline":
        # overridden to also move the first frame generator pipeline if it exists
        if self.first_frame_gen_pipeline is not None:
            self.first_frame_gen_pipeline.to(device, **kwargs)
        super().to(device, **kwargs)
        return self

    @classmethod
    def from_pretrained(
        cls, model_id: str, mode: str = "hybrid", cache_dir: str = None, **kwargs: Any
    ) -> "NeodragonPipeline":
        _validate_mode(mode)

        # download the model from HuggingFace and get the local path
        assert (
            cache_dir is not None
        ), "cache_dir must be specified to download the model from HuggingFace"
        local_model_path = snapshot_download(model_id, cache_dir=cache_dir)

        # Load all components:
        text_encoder_bundle = TextEncoderBundle.from_pretrained(local_model_path, **kwargs)
        # NOTE: the context adapter and DIT differ between monolithic and hybrid modes
        context_adapter_id = (
            CONTEXT_ADAPTER_ID if mode == "hybrid" else MULTISTEP_CONTEXT_ADAPTER_ID
        )
        context_adapter = ContextAdapter.from_pretrained(
            f"{local_model_path}/{context_adapter_id}", **kwargs
        )
        dit_id = DIT_ID if mode == "hybrid" else MULTISTEP_DIT_ID
        dit = PyramidMMDiT.from_pretrained(f"{local_model_path}/{dit_id}", **kwargs)
        vae = AsymmetricCausalVideoVAE.from_pretrained(f"{local_model_path}/{VAE_ID}", **kwargs)
        scheduler = PyramidFlowMatchEulerDiscreteScheduler()

        return cls(
            text_encoder_bundle=text_encoder_bundle,
            context_adapter=context_adapter,
            dit=dit,
            vae=vae,
            scheduler=scheduler,
            model_path=local_model_path,
            mode=mode,
        )

    def __call__(
        self,
        prompt: str,
        image: Optional[Image] = None,
        prompt_modifier: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        profile: bool = False,
        **kwargs: Any,
    ) -> Union[torch.FloatTensor, List[Image]]:
        # construct the generation configuration based on the selected mode
        mode_conf = self.config.gen_confs[self.mode]
        # common generation config overrides mode-specific config
        gen_conf = {**self.config.gen_confs["common"], **mode_conf}
        # finally everything overridden by kwargs
        gen_conf = {**gen_conf, **kwargs}

        if self.mode == "monolithic":
            return generate(
                self.text_encoder_bundle,
                self.dit,
                self.context_adapter,
                self.vae,
                self.scheduler,
                frames_per_unit=self.config.frames_per_unit,
                num_stages=len(self.config.stages),
                prompt=prompt,
                image=image,
                device=self.device,
                dtype=self.dtype,
                prompt_modifier=prompt_modifier,
                negative_prompt=negative_prompt,
                profile=profile,
                **gen_conf,
            )
        else:
            return generate_hybrid(
                self.first_frame_gen_pipeline,
                self.text_encoder_bundle,
                self.dit,
                self.context_adapter,
                self.vae,
                self.scheduler,
                frames_per_unit=self.config.frames_per_unit,
                num_stages=len(self.config.stages),
                prompt=prompt,
                device=self.device,
                dtype=self.dtype,
                prompt_modifier=prompt_modifier,
                negative_prompt=negative_prompt,
                profile=profile,
                **gen_conf,
            )
