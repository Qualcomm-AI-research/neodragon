# fmt: off
#!/usr/bin/env python3

# Copyright (c) 2026 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import os
from concurrent import futures
from pathlib import Path
from typing import Tuple

import torch
from diffusers.utils import export_to_video
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm

from neodragon import PIPELINE_MODES, NeodragonPipeline
from neodragon.safety_checker import (
    check_generated_video_safety,
    load_stable_diffusion_safety_checker,
)
from neodragon.utils import (
    get_torch_dtype,
    init_distributed_mode,
    is_dist_avail_and_initialized,
)

FPS = 24
MAX_VIDEO_WRITER_WORKERS = 16
HF_MODEL_ID = "karnewar/Neodragon"


class _PromptsDataset(Dataset):
    def __init__(
        self,
        prompts_file: str,
        num_samples: int = 5,
        max_num_prompts: int = -1,  # used for debugging
    ) -> None:
        super().__init__()

        self.prompts_file = prompts_file
        self.num_samples = num_samples
        self.max_num_prompts = max_num_prompts

        # Read the prompts file:
        with open(self.prompts_file, "r") as file:
            lines = file.readlines()
        self.all_prompts = [line.strip() for line in lines]

        # filtering based on max_num_prompts:
        if self.max_num_prompts > 0:
            self.all_prompts = self.all_prompts[: self.max_num_prompts]

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        prompt = self.all_prompts[idx]
        # hacky but allows to couple num_samples with prompt in dataloader
        return (prompt, self.num_samples)

    def __len__(self):
        return len(self.all_prompts)


def _build_data_loader(
    prompts_file: str,
    num_samples: int,
    max_num_prompts: int,
    world_size: int = 1,
    rank: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = _PromptsDataset(
        prompts_file=prompts_file,
        num_samples=num_samples,
        max_num_prompts=max_num_prompts,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    prompts_dataloader = DataLoader(
        dataset,
        batch_size=1,  # one prompt per GPU
        num_workers=6,  # number of worker processes for data loading
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        prefetch_factor=2,
    )

    return prompts_dataloader


def get_args():
    parser = argparse.ArgumentParser("Neodragon Video Generation Script", add_help=True)

    parser.add_argument(
        "--model_dtype",
        default="bf16",
        type=str,
        help="The Model Dtype: bf16 or fp16 or fp32. bf16 is default for fast inference",
    )
    # fmt: off
    parser.add_argument(
        "--mode",
        default="hybrid",
        type=str,
        choices=PIPELINE_MODES,
        help="The Neodragon Pipeline Mode | Choices: [" + ", ".join(PIPELINE_MODES) + "]",
    )
    # fmt: on
    parser.add_argument(
        "--local_cache_folder",
        type=str,
        default="./models",
        help="Path to a local cache folder for storing downloaded models from HuggingFace.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./prompts/showcase_prompts.txt",
        help="Path to a `.txt` file with prompts (one per line)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="The number of videos to be generated per prompt",
    )
    parser.add_argument(
        "--max_num_prompts",
        type=int,
        default=-1,
        help="The number of maximum prompts to use (Helpful for Debugging)",
    )
    parser.add_argument(
        "--use_cinematic_prompt_modifier",
        action="store_true",
        help="Whether to use the cinematic prompt modifier or not",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile the generation time or not",
    )

    parser.add_argument("--height", type=int, default=320, help="The video height")
    parser.add_argument("--width", type=int, default=512, help="The video width")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="The number of frames in the generated video",
    )
    parser.add_argument(  # Don't touch this!
        "--fps", type=int, default=FPS, help="fps of the exported videos"
    )

    parser.add_argument(
        "--output_video_folder",
        default="./output/neodragon_videos",
        type=str,
        help="The path where the generated videos should be saved",
    )

    # Safety Checker arguments:
    parser.add_argument(
        "--safety_checker_model_id",
        type=str,
        default="CompVis/stable-diffusion-safety-checker",
        help="Hugging Face model ID for the Stable Diffusion safety checker.",
    )

    parser.add_argument(
        "--safety_check_num_frames",
        type=int,
        default=1,
        help="Number of generated video frames to sample for output safety checking.",
    )

    parser.add_argument(
        "--disable_safety_checker",
        action="store_true",
        help="Whether to disable the Stable Diffusion safety checker.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    init_distributed_mode(args)

    try:
        world_size = args.world_size
        rank = args.rank
        local_gpu = args.gpu
    except AttributeError:
        world_size = 1
        rank = 0
        local_gpu = 0

    device = torch.device(f"cuda:{local_gpu}" if torch.cuda.is_available() else "cpu")
    torch_dtype = get_torch_dtype(args.model_dtype)

    # Load the pipeline
    neodragon_pipeline = NeodragonPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch_dtype,
        mode=args.mode,
        cache_dir=args.local_cache_folder,
    ).to(device)
    print(f"Loaded Neodragon Pipeline in {args.mode} mode.")

    # Load the safety checker:
    safety_checker, safety_feature_extractor = None, None
    if not args.disable_safety_checker:
        safety_checker, safety_feature_extractor = load_stable_diffusion_safety_checker(
            model_id=args.safety_checker_model_id,
            device=device,
            torch_dtype=torch_dtype,
            cache_dir=args.local_cache_folder,
        )
        print(f"Loaded Stable Diffusion Safety Checker.")

    # Build the distributed data loader:
    dataloader = _build_data_loader(
        prompts_file=args.prompts_file,
        num_samples=args.num_samples,
        max_num_prompts=args.max_num_prompts,
        world_size=world_size,
        rank=rank,
    )

    if is_dist_avail_and_initialized():
        # synchronize all processes / GPUs before starting inference
        torch.distributed.barrier()

    prompt_modifier = (
        ", cinematic, realistic textures, high detail, natural colours"
        if args.use_cinematic_prompt_modifier
        else ", hyper quality, Ultra HD, 8K"
    )

    # Video generation loop:
    task_queue = []
    with futures.ThreadPoolExecutor(max_workers=MAX_VIDEO_WRITER_WORKERS) as executor:
        for datum in tqdm(dataloader):
            prompt, num_samples = datum[0][0], datum[1].item()

            with (
                torch.no_grad(),
                torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype),
            ):
                for sample_id in range(num_samples):
                    output_video_path = Path(args.output_video_folder) / f"{prompt}-{sample_id}.mp4"

                    if output_video_path.exists() and output_video_path.stat().st_size > 0:
                        continue

                    # generate the video:
                    generated_video = neodragon_pipeline(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        prompt_modifier=prompt_modifier,
                        profile=args.profile,
                    )

                    # Safety check the generated video:
                    if not args.disable_safety_checker:
                        is_unsafe, frame_safety_flags = check_generated_video_safety(
                            generated_video=generated_video,
                            safety_checker=safety_checker,
                            feature_extractor=safety_feature_extractor,
                            device=device,
                            torch_dtype=torch_dtype,
                            num_frames=args.safety_check_num_frames,
                        )

                        if is_unsafe:
                            message = (
                                f"Generated video rejected by StableDiffusionSafetyChecker. "
                                f"prompt={prompt!r}, frame_safety_flags={frame_safety_flags}"
                            )
                            raise RuntimeError(message)

                    # If video passes safety check submit it for exporting
                    task_queue.append(
                        executor.submit(
                            export_to_video,
                            generated_video,
                            output_video_path,
                            fps=args.fps,
                        )
                    )

        for future in futures.as_completed(task_queue):
            # Consume completions while the executor is still in scope
            try:
                future.result()  # raises if export_to_video failed
            except Exception as e:
                # Log and (optionally) decide to abort or continue
                print(f"[export error] {e}")

    # again synchronize all processes / GPUs before exiting
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()

    print(f"Generation complete! check path: {args.output_video_folder}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    parsed_args = get_args()
    if not Path(parsed_args.output_video_folder).exists():
        Path(parsed_args.output_video_folder).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
