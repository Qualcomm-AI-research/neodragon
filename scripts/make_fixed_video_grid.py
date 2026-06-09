#!/usr/bin/env python3

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse

from neodragon.utils.visualisation_utils import make_fixed_video_grid


def get_args():
    parser = argparse.ArgumentParser(
        "Helper Script For Making Fixed Video Grid", add_help=True
    )

    parser.add_argument(
        "--source_path",
        default=None,
        required=True,
        type=str,
        help="The path to the folder containing videos to be arranged in a grid",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        required=False,
        type=str,
        help="The path where the fixed video grid should be saved",
    )
    parser.add_argument(
        "--output_file_name",
        default="video_grid.mp4",
        required=False,
        type=str,
        help="The file name for the fixed video grid",
    )
    parser.add_argument(
        "--max_videos",
        default=48,
        required=False,
        type=int,
        help="The maximum number of videos to include in the grid",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    print("Making fixed video grid...")
    make_fixed_video_grid(
        source_path=args.source_path,
        output_path=args.output_path,
        output_file_name=args.output_file_name,
        max_videos=args.max_videos,
    )
    print("Done!")


if __name__ == "__main__":
    parsed_args = get_args()
    main(parsed_args)
