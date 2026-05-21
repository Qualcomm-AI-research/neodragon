# Copyright (c) 2026 Qualcomm Technologies, Inc.
# All Rights Reserved.

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers.utils.export_utils import export_to_video
from einops import rearrange
from torchvision.io import read_video
from torchvision.utils import make_grid


def make_fixed_video_grid(
    source_path: str,
    output_path: Optional[str] = None,
    output_file_name: str = "video_grid.mp4",
    max_videos: int = 48,
) -> None:
    assert max_videos > 0, "Provided negative number of max videos"

    source_path = Path(source_path)
    if not source_path.exists():
        raise RuntimeError(f"Provided source path {str(source_path)} does not exist")
    if output_path is None:
        output_path = source_path / "video_grid"
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # setup random with a fixed seed
    RANDOM_SEED = 16041969
    random.seed(RANDOM_SEED)

    # make and save the video grid
    video_paths = sorted(source_path.glob("*.mp4"))
    if len(video_paths) == 0:
        raise RuntimeError(f"No videos found under {str(source_path)}")
    random.shuffle(video_paths)
    video_paths = video_paths[: min(len(video_paths), max_videos)]

    # only use the vFrames with [0]
    videos = [read_video(str(vid_path), pts_unit="sec")[0] for vid_path in video_paths]
    videos = rearrange(torch.stack(videos, dim=0), "b t h w c -> t b c h w")

    if int(np.sqrt(videos.shape[1])) ** 2 == videos.shape[1]:
        num_columns = int(np.ceil(np.sqrt(int(videos.shape[1]))))
    else:
        # prefer more columns than rows
        num_columns = int(np.ceil(np.sqrt(int(videos.shape[1])))) + 1
    video_grid = torch.stack([make_grid(frame, nrow=num_columns, padding=0) for frame in videos])
    video_grid = video_grid.float() / 255.0
    video_grid = rearrange(video_grid, "t c h w -> t h w c")
    export_to_video(video_grid.cpu().numpy(), str(output_path / output_file_name), fps=24)
