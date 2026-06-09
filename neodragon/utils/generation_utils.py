# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image
from tqdm.auto import tqdm

from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
from neodragon.context_adapter import ContextAdapter
from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline
from neodragon.pyramid_mmdit import PyramidMMDiT
from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
from neodragon.text_encoder_bundle import TextEncoderBundle
from neodragon.utils import Timer

VAE_SCALE_FACTOR = 0.5430
VAE_SHIFT_FACTOR = 0.1490
VAE_VIDEO_SCALE_FACTOR = 0.3031
VAE_VIDEO_SHIFT_FACTOR = -0.2343

DEFAULT_PROMPT_MODIFIER = (
    ", cinematic, realistic textures, high detail, natural colours"
)
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon style, worst quality, low quality, blurry, absolute black, "
    "absolute white, low res, extra limbs, extra digits, misplaced objects, "
    "mutated anatomy, monochrome, horror"
)


# ------------------------------------------------------------------------------------------------------ #
# The main proposal of the Pyramidal-Flow paper! This is the key part of the pyramid-flow                #
# This allows the model to add corrective gausian noise (roll back time-step at higher resolution)       #
# And, thus be able to follow the same flow-matching trajectory as that at the highest reoslution!       #
# This is coarse-to-fine mutation generative model at it's best!                                         #
# Reference: Eqn 15 of the paper https://arxiv.org/pdf/2410.05954                                        #
# ------------------------------------------------------------------------------------------------------ #


def _upsample_pyramidal_latent(
    latents: torch.FloatTensor,
    orig_sigma: float,
    gamma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.FloatTensor:
    # Upsample the latents by a factor of 2
    t = latents.shape[2]
    latents = rearrange(latents, "b c t h w -> (b t) c h w")
    latents = torch.nn.functional.interpolate(latents, scale_factor=2, mode="nearest")
    latents = rearrange(latents, "(b t) c h w -> b c t h w", t=t)

    # Fix the stage sigma
    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - orig_sigma) + orig_sigma)
    beta = alpha * (1 - orig_sigma) / math.sqrt(gamma)

    # add the corrective noise
    bs, ch, temp, height, width = latents.shape
    noise = _sample_block_noise(gamma, bs, ch, temp, height, width)
    noise = noise.to(device=device, dtype=dtype)
    latents = alpha * latents + beta * noise  # To fix the block artifact

    return latents


# ------------------------------------------------------------------------------------------------------ #


def _prepare_latent_noise(
    batch_size: int,
    num_channels_latents: int,
    temp: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.FloatTensor:
    shape = (
        batch_size,
        num_channels_latents,
        int(temp),
        int(height),
        int(width),
    )
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return latents


def _sample_block_noise(
    gamma: float, bs: int, ch: int, temp: int, height: int, width: int
) -> torch.FloatTensor:
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(4), torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma
    )
    block_number = bs * ch * temp * (height // 2) * (width // 2)
    noise = torch.stack([dist.sample() for _ in range(block_number)])
    noise = rearrange(
        noise,
        "(b c t h w) (p q) -> b c t (h p) (w q)",
        b=bs,
        c=ch,
        t=temp,
        h=height // 2,
        w=width // 2,
        p=2,
        q=2,
    )
    return noise


@torch.no_grad()
def _get_pyramid_latent(x, num_stages: int) -> List[torch.FloatTensor]:
    # x is the origin vae latent
    vae_latent_list = []
    vae_latent_list.append(x)

    temp, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
    for _ in range(num_stages - 1):
        height //= 2
        width //= 2
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear")
        x = rearrange(x, "(b t) c h w -> b c t h w", t=temp)
        vae_latent_list.append(x)

    vae_latent_list = list(reversed(vae_latent_list))
    return vae_latent_list


def _numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    images = np.clip((images * 127.5) + 127.5, 0, 255).astype(np.uint8)
    if images.ndim == 3:
        images = images[None, ...]

    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def _pil_to_numpy(images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
    if isinstance(images, Image.Image):
        images = [images]
    numpy_images = [np.array(image) for image in images]
    numpy_images = [
        (image.astype(np.float32) / 255.0) * 2 - 1 for image in numpy_images
    ]
    return np.stack(numpy_images)


def _downsample_noise_2x(latents: torch.FloatTensor, times: int) -> torch.FloatTensor:
    _, _, num_frames, height, width = latents.shape
    latents = rearrange(latents, "b c t h w -> (b t) c h w")
    for _ in range(times):
        height //= 2
        width //= 2
        # the multiplication by 2.0 is to keep the variance consistent after downsampling
        latents = (
            torch.nn.functional.interpolate(
                latents, size=(height, width), mode="bilinear"
            )
            * 2.0
        )
    latents = rearrange(latents, "(b t) c h w -> b c t h w", t=num_frames)
    return latents


def _prepare_past_condition_latents(
    generated_latents_list: List[torch.FloatTensor],
    num_stages: int,
    do_classifier_free_guidance: bool = True,
) -> List[List[torch.FloatTensor]]:
    if len(generated_latents_list) == 0:
        return [[] for _ in range(num_stages)]

    frames_per_unit = generated_latents_list[0].shape[2]
    unit_index = len(generated_latents_list)
    history_latents_tensor = torch.cat(generated_latents_list, dim=2)  # temporal dim
    history_latents_pyramid = _get_pyramid_latent(history_latents_tensor, num_stages)

    past_condition_latents = []
    for stage in range(num_stages):
        # start from the last current stage latent
        last_cond_latent = history_latents_pyramid[stage][:, :, -frames_per_unit:, ...]
        stage_input = [last_cond_latent]

        # pad the past clean latents
        cur_unit_num = unit_index
        cur_stage = stage
        cur_unit_ptx = 1
        while cur_unit_ptx < cur_unit_num:
            cur_stage = max(cur_stage - 1, 0)
            if cur_stage == 0:
                break
            cur_unit_ptx += 1
            chunk_begin = -(cur_unit_ptx * frames_per_unit)
            chunk_end = -((cur_unit_ptx - 1) * frames_per_unit)
            cond_latents = history_latents_pyramid[cur_stage][
                :, :, chunk_begin:chunk_end, ...
            ]
            stage_input.append(cond_latents)
        if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
            cond_latents = history_latents_pyramid[0][
                :, :, : -(cur_unit_ptx * frames_per_unit), ...
            ]
            stage_input.append(cond_latents)

        stage_input = list(reversed(stage_input))
        past_condition_latents.append(stage_input)

    if do_classifier_free_guidance:
        past_condition_latents = [
            [torch.cat([lat] * 2) for lat in stage_latents]
            for stage_latents in past_condition_latents
        ]

    return past_condition_latents


@torch.no_grad()
def _decode_latent(
    vae: AsymmetricCausalVideoVAE,
    latents: torch.FloatTensor,
    return_tensor: bool = False,
) -> Union[List[Image.Image], torch.FloatTensor]:
    if latents.shape[2] == 1:
        latents = (latents / VAE_SCALE_FACTOR) + VAE_SHIFT_FACTOR
    else:
        latents[:, :, :1] = (latents[:, :, :1] / VAE_SCALE_FACTOR) + VAE_SHIFT_FACTOR
        latents[:, :, 1:] = (
            latents[:, :, 1:] / VAE_VIDEO_SCALE_FACTOR
        ) + VAE_VIDEO_SHIFT_FACTOR

    video = vae.decode(latents).sample
    video = rearrange(video, "B C T H W -> (B T) H W C")

    if return_tensor:
        return video

    # convert to PIL
    video = video.float().cpu().numpy()
    video = _numpy_to_pil(video)
    return video


@torch.no_grad()
def _generate_one_unit(
    scheduler: PyramidFlowMatchEulerDiscreteScheduler,
    dit: PyramidMMDiT,
    num_stages: int,
    latents: torch.FloatTensor,
    past_conditions: List[List[torch.FloatTensor]],
    prompt_embeds: torch.FloatTensor,
    prompt_attention_mask: torch.FloatTensor,
    pooled_prompt_embeds: torch.FloatTensor,
    num_inference_steps: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    do_classifier_free_guidance: bool = True,
    guidance_scale: float = 7.0,  # for first frame
    video_guidance_scale: float = 5.0,  # for rest of the frames
    profile: bool = False,
    show_denoising: bool = False,
) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
    intermed_latents = []  # to store all intermediate latents
    is_first_frame = all(len(cond) == 0 for cond in past_conditions)

    for stage in range(num_stages):
        timesteps = scheduler.get_stage_timesteps(
            num_inference_steps[stage], stage, device=device
        )
        sigmas = scheduler.get_stage_sigmas(
            num_inference_steps[stage], stage, device=device
        )

        if stage > 0:
            with Timer(f"Upsample from stage {stage-1} to stage {stage}", profile):
                latents = _upsample_pyramidal_latent(
                    latents=latents,
                    # Note the (1 - orig_sigma) here to get the correct orig sigma
                    orig_sigma=(1 - scheduler.orig_start_sigmas[stage]),
                    gamma=scheduler.config.gamma,
                    device=device,
                    dtype=dtype,
                )

        with Timer(
            f"stage {stage} Latent [{latents.shape[-2]} x {latents.shape[-1]}] total time",
            profile,
        ):
            for t in range(len(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )

                timestep = (
                    timesteps[t]
                    .expand(latent_model_input.shape[0])
                    .to(latent_model_input.dtype)
                )
                sigma = sigmas[t].to(latent_model_input.dtype)
                sigma_next = sigmas[t + 1].to(latent_model_input.dtype)

                # DiT forward pass (Obtain the Predicted Flow for generation)
                with Timer(
                    f"DIT step {t} Latent shape [{latents.shape[-3]} x {latents.shape[-2]} x {latents.shape[-1]}]",
                    profile,
                ):
                    if profile:
                        print(
                            f"Actual DiT input tokens: {[[la.shape for la in lat] for lat in [latent_model_input]]}"
                        )
                        print(f"Encoder hidden states: {prompt_embeds.shape}")
                        print(f"Encoder attention mask: {prompt_attention_mask.shape}")
                        print(f"Pooled projections: {pooled_prompt_embeds.shape}")

                    latent_model_input = past_conditions[stage] + [latent_model_input]
                    noise_pred = dit(
                        sample=[latent_model_input],
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        timestep_ratio=timestep,
                    )

                noise_pred = noise_pred[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if is_first_frame:
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred = noise_pred_uncond + video_guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                # dnoising step
                latents = scheduler.step(
                    model_output=noise_pred,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    sample=latents,
                ).prev_sample

                intermed_latents.append(latents)

    if show_denoising:
        return intermed_latents[-1], intermed_latents

    return intermed_latents[-1]


@torch.no_grad()
def generate_hybrid(
    first_frame_gen_pipeline: SSD1B_FirstFrameGeneratorPipeline,
    text_encoder_bundle: TextEncoderBundle,
    dit: PyramidMMDiT,
    context_adapter: ContextAdapter,
    vae: AsymmetricCausalVideoVAE,
    scheduler: PyramidFlowMatchEulerDiscreteScheduler,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = 320,
    width: Optional[int] = 512,
    num_frames: int = 49,
    num_inference_steps: Optional[Union[int, List[int]]] = [1, 1, 1],
    video_num_inference_steps: Optional[Union[int, List[int]]] = [1, 1, 1],
    do_classifier_free_guidance: bool = False,
    guidance_scale: float = 0.0,
    video_guidance_scale: float = 0.0,
    prompt_modifier: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    frames_per_unit: int = 1,
    num_stages: int = 3,
    output_type: Optional[str] = "pil",
    profile: bool = False,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    prompt_modifier = (
        DEFAULT_PROMPT_MODIFIER if prompt_modifier is None else prompt_modifier
    )
    negative_prompt = (
        DEFAULT_NEGATIVE_PROMPT if negative_prompt is None else negative_prompt
    )

    with Timer("First Frame Generation", profile):
        first_frame = first_frame_gen_pipeline(
            prompt=prompt + prompt_modifier,
            num_images_per_prompt=1,
        ).images[0]

    return generate(
        text_encoder_bundle=text_encoder_bundle,
        dit=dit,
        context_adapter=context_adapter,
        vae=vae,
        scheduler=scheduler,
        prompt=prompt,
        image=first_frame,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        video_num_inference_steps=video_num_inference_steps,
        do_classifier_free_guidance=do_classifier_free_guidance,
        guidance_scale=guidance_scale,
        video_guidance_scale=video_guidance_scale,
        prompt_modifier=prompt_modifier,
        negative_prompt=negative_prompt,
        frames_per_unit=frames_per_unit,
        num_stages=num_stages,
        output_type=output_type,
        profile=profile,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def generate(
    text_encoder_bundle: TextEncoderBundle,
    dit: PyramidMMDiT,
    context_adapter: ContextAdapter,
    vae: AsymmetricCausalVideoVAE,
    scheduler: PyramidFlowMatchEulerDiscreteScheduler,
    prompt: str = None,
    image: Optional[Union[Image.Image, List[Image.Image]]] = None,
    height: Optional[int] = 320,
    width: Optional[int] = 512,
    num_frames: int = 49,
    num_inference_steps: Optional[Union[int, List[int]]] = [20, 20, 20],
    video_num_inference_steps: Optional[Union[int, List[int]]] = [10, 10, 10],
    frames_per_unit: int = 1,
    num_stages: int = 3,
    do_classifier_free_guidance: bool = True,
    guidance_scale: float = 7.0,
    video_guidance_scale: float = 5.0,
    prompt_modifier: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    output_type: Optional[str] = "pil",
    profile: bool = False,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    prompt_modifier = (
        DEFAULT_PROMPT_MODIFIER if prompt_modifier is None else prompt_modifier
    )
    negative_prompt = (
        DEFAULT_NEGATIVE_PROMPT if negative_prompt is None else negative_prompt
    )

    num_latent_frames = ((num_frames - 1) // vae.config.temporal_downsample_scale) + 1
    assert (
        num_latent_frames - 1
    ) % frames_per_unit == 0, "The total frames should be divided by `frames_per_unit`"
    num_latent_units = 1 + (num_latent_frames - 1) // frames_per_unit

    if isinstance(num_inference_steps, int):
        num_inference_steps = [num_inference_steps] * num_stages

    if isinstance(video_num_inference_steps, int):
        video_num_inference_steps = [video_num_inference_steps] * num_stages

    with Timer("Text Prompt processing", profile):
        prompt = prompt + prompt_modifier
        negative_prompt = negative_prompt or ""

        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = (
            text_encoder_bundle(prompt, device)
        )

        if do_classifier_free_guidance:
            (
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                negative_pooled_prompt_embeds,
            ) = text_encoder_bundle(negative_prompt, device)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

    with Timer("Context Adaptation", profile):
        prompt_embeds = context_adapter(prompt_embeds)

    with Timer("Initial Random Noise Preparation", profile):
        dit_config = (
            dit.module.config
            if isinstance(dit, torch.nn.parallel.distributed.DistributedDataParallel)
            else dit.config
        )
        num_channels_latents = dit_config.in_channels
        latents = _prepare_latent_noise(
            1,  # batch size
            num_channels_latents,
            num_latent_frames,
            height // vae.config.spatial_downsample_scale,
            width // vae.config.spatial_downsample_scale,
            prompt_embeds.dtype,
            device,
        )

        # We start with the lowest resolution latent_noise
        latents = _downsample_noise_2x(latents, num_stages - 1)

    with Timer("Optional First Frame Processing", profile):
        if image is not None:
            print("Using provided image as the first frame ...")
            # If image is not None, we can use it as the first frame to guide the generation
            image = image.resize((width, height), resample=Image.LANCZOS)
            image = _pil_to_numpy(image)
            image = torch.from_numpy(image).to(device=device, dtype=dtype)
            image = rearrange(image, "(b t) h w c -> b c t h w", b=1, t=1)
            image_latent = vae.encode(image).latent_dist.sample()
            image_latent = rearrange(image_latent, "b c t h w -> b c t h w")
            # NOTE: Don't forget :) to scale and shift the latent!
            image_latent = (image_latent - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR
            generated_latents_list = [image_latent]
        else:
            # Otherwise, we generate all frames from scratch
            print("No initial frame provided, generating all frames ...")
            generated_latents_list = []

    with Timer("AutoRegressive Diffusion hybrid denoising", profile):
        start_unit = len(generated_latents_list)
        for unit_index in tqdm(range(start_unit, num_latent_units)):
            with Timer(f"Latent frame {unit_index} denoising", profile):
                # prepare the past condition latents given the generated latents so far
                past_condition_latents = _prepare_past_condition_latents(
                    generated_latents_list=generated_latents_list,
                    num_stages=num_stages,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )

                # now, we can denoise the current unit
                chunk_start = unit_index * frames_per_unit
                chunk_end = (unit_index + 1) * frames_per_unit
                intermed_latents = _generate_one_unit(
                    scheduler=scheduler,
                    dit=dit,
                    num_stages=num_stages,
                    latents=latents[:, :, chunk_start:chunk_end, ...],
                    past_conditions=past_condition_latents,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=(
                        num_inference_steps
                        if unit_index == 0
                        else video_num_inference_steps
                    ),
                    device=device,
                    dtype=dtype,
                    profile=profile,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guidance_scale=guidance_scale,
                    video_guidance_scale=video_guidance_scale,
                )
                generated_latents_list.append(intermed_latents)
        generated_latents = torch.cat(generated_latents_list, dim=2)

    with Timer("Decoding", profile):
        if output_type == "latent":
            generated_video = generated_latents
        else:
            generated_video = _decode_latent(vae, generated_latents)

    return generated_video
