# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-License
#
# Authors: Guojin Chen (work done during internship at NVIDIA), Haoyu Yang
#
# Project: DiffOPC — Differentiable OPC
# Paper:   https://dl.acm.org/doi/10.1145/3676536.3676764
#
# NVIDIA CORPORATION and its affiliates retain all intellectual property and
# proprietary rights in and to this software and related documentation. Any
# use, reproduction, or distribution is subject to the terms of the NVIDIA
# License (see LICENSE in the project root). The work may be used only for
# non-commercial research or evaluation, except by NVIDIA Corporation and
# its affiliates.

from typing import List

import torch
from torch.utils.data import Dataset

import src.data.loaders.glp_seg as glp_seg
from src.data.loaders.segments import SegmentsInitTorch


class Iccad13MultiScale(Dataset):
    def __init__(
        self,
        data_dir: str,
        tile_size_x: int,
        tile_size_y: int,
        offset_x: int,
        offset_y: int,
        start_idx: int,
        end_idx: int,
        sraf_forbiddens: List[int],
        seg_lengths: List[int],
        down_scales: List[int],
        device: torch.device,
        scale_strs: List[str] = ["low", "mid", "high"],
    ):
        self.data_dir = data_dir
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.sraf_forbiddens = sraf_forbiddens
        self.down_scales = down_scales
        self.scale_strs = scale_strs
        self.seg_lengths = seg_lengths
        self.device = device

    def __len__(self):
        return self.end_idx - self.start_idx + 1

    def __getitem__(self, index):
        data_idx = index + self.start_idx
        assert data_idx <= self.end_idx, f"Index {index} out of bounds"
        glp_path = f"{self.data_dir}/M1_test{data_idx}.glp"
        data_obj = {}
        for scale_str, seg_length, sraf_forbidden, down_scale in zip(
            self.scale_strs, self.seg_lengths, self.sraf_forbiddens, self.down_scales
        ):
            design = glp_seg.Design(glp_path, down=down_scale)
            # when run, already centered.
            target, edge_params, metadata = SegmentsInitTorch().run(
                design,
                self.tile_size_x // down_scale,
                self.tile_size_y // down_scale,
                self.offset_x // down_scale,
                self.offset_y // down_scale,
                seg_length,
                sraf_forbidden,
                self.device,
            )
            data_obj[scale_str] = (target, edge_params, metadata, data_idx)
        return data_obj


class Iccad13MultiScaleSingle(Dataset):
    def __init__(
        self,
        data_dir: str,
        tile_size_x: int,
        tile_size_y: int,
        offset_x: int,
        offset_y: int,
        data_idx: int,
        sraf_forbiddens: List[int],
        seg_lengths: List[int],
        down_scales: List[int],
        device: torch.device,
        scale_strs: List[str] = ["low", "mid", "high"],
    ):
        self.data_dir = data_dir
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.sraf_forbiddens = sraf_forbiddens
        self.data_idx = data_idx
        self.down_scales = down_scales
        self.scale_strs = scale_strs
        self.seg_lengths = seg_lengths
        self.device = device

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # assert index == 0, f"Index {index} out of bounds"
        data_idx = index + self.data_idx
        glp_path = f"{self.data_dir}/M1_test{data_idx}.glp"
        data_obj = {}
        for scale_str, seg_length, sraf_forbidden, down_scale in zip(
            self.scale_strs, self.seg_lengths, self.sraf_forbiddens, self.down_scales
        ):
            design = glp_seg.Design(glp_path, down=down_scale)
            # when run, already centered.
            target, edge_params, metadata = SegmentsInitTorch().run(
                design,
                self.tile_size_x // down_scale,
                self.tile_size_y // down_scale,
                self.offset_x // down_scale,
                self.offset_y // down_scale,
                seg_length,
                sraf_forbidden,
                self.device,
            )
            data_obj[scale_str] = (target, edge_params, metadata, data_idx)
        return data_obj
