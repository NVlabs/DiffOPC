#!/usr/bin/bash
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


python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python


min_areas=(5 10 15 20 30 40 50)
min_whs=(1 2 3 4 5 6 7 8 9 10)

for min_area in "${min_areas[@]}"; do
    for min_wh in "${min_whs[@]}"; do
        $python src/mrc/mrc.py min_area=$min_area min_wh=$min_wh exp_name=mrc_marea_${min_area}
    done
done
