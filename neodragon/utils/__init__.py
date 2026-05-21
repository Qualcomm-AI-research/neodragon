# Copyright (c) 2026 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import datetime
import os
import time
from typing import Optional

import torch
import torch.distributed as dist


def _setup_for_distributed(is_master: bool) -> None:
    # Disables printing to stdout/stderr for non-master processes
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args: argparse.Namespace, init_pytorch_ddp: bool = True) -> None:
    if int(os.getenv("OMPI_COMM_WORLD_SIZE", "0")) > 0:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    args.dist_backend = "nccl"
    args.dist_url = "env://"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu),
        flush=True,
    )

    if init_pytorch_ddp:
        # Init DDP Group, for script without using accelerate framework
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(days=365),
        )
        torch.distributed.barrier()
        _setup_for_distributed(args.rank == 0)


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class Timer:
    def __init__(self, title: Optional[str] = None, profile: bool = True) -> None:
        self.title = title if title else ""
        self.profile = profile

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, _exc_type: type, _exc_val: Exception, _exc_tb: type) -> None:
        end_time = time.time()
        total_time = end_time - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        if self.profile:
            print(
                f"[{self.title}]: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.5f} seconds."
            )


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    return torch_dtype
