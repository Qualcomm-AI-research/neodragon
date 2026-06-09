# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from transformers import CLIPImageProcessor


def load_stable_diffusion_safety_checker(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    cache_dir: str,
):
    feature_extractor = CLIPImageProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    ).to(device)

    safety_checker.eval()

    return safety_checker, feature_extractor


def _to_pil_image(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if torch.is_tensor(frame):
        frame = frame.detach().cpu()

        # Accept CHW or HWC.
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            frame = frame.permute(1, 2, 0)

        frame = frame.float().numpy()

    if isinstance(frame, np.ndarray):
        # Convert possible [0, 1] float image to [0, 255] uint8.
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255).round().astype(np.uint8)

        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        if frame.shape[-1] == 4:
            frame = frame[..., :3]

        return Image.fromarray(frame).convert("RGB")

    raise TypeError(f"Unsupported frame type for safety checking: {type(frame)}")


def _sample_video_frames(video, num_frames: int):
    """
    Handles common export_to_video-compatible outputs:
      - list[PIL.Image]
      - list[np.ndarray]
      - torch.Tensor with shape TCHW, THWC, BTHWC, or BTCHW
      - np.ndarray with shape THWC or BTHWC
    """
    if torch.is_tensor(video):
        video = video.detach().cpu()

        if video.ndim == 5:
            # Take first batch item.
            video = video[0]

        frames = list(video)

    elif isinstance(video, np.ndarray):
        if video.ndim == 5:
            video = video[0]

        frames = list(video)

    elif isinstance(video, list):
        frames = video

    else:
        raise TypeError(f"Unsupported video type for safety checking: {type(video)}")

    if len(frames) == 0:
        return []

    if num_frames <= 0 or num_frames >= len(frames):
        sampled_frames = frames
    else:
        indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        sampled_frames = [frames[i] for i in indices]

    return [_to_pil_image(frame) for frame in sampled_frames]


@torch.no_grad()
def check_generated_video_safety(
    generated_video,
    safety_checker: StableDiffusionSafetyChecker,
    feature_extractor,
    device: torch.device,
    torch_dtype: torch.dtype,
    num_frames: int,
):
    sampled_frames = _sample_video_frames(
        generated_video,
        num_frames=num_frames,
    )

    if len(sampled_frames) == 0:
        return False, []

    safety_checker_input = feature_extractor(
        sampled_frames,
        return_tensors="pt",
    ).to(device)

    # StableDiffusionSafetyChecker expects images as numpy float arrays in [0, 1].
    images_np = np.stack(
        [np.asarray(frame).astype(np.float32) / 255.0 for frame in sampled_frames],
        axis=0,
    )

    _, has_unsafe_concept = safety_checker(
        images=images_np,
        clip_input=safety_checker_input.pixel_values.to(
            device=device,
            dtype=torch_dtype,
        ),
    )

    has_unsafe_concept = list(has_unsafe_concept)
    is_unsafe = any(has_unsafe_concept)

    return is_unsafe, has_unsafe_concept
