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

data_idxs=(1 2 3 4 5 6 7 8 9 10)

# res=high

# for data_idx in "${data_idxs[@]}"; do
#     $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=srafgen
#     mv tmp/grad_2048x2048 tmp/grad_2048x2048_srafgen$data_idx
# done
# # $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=true logger.aim.experiment=srafgen




# res=mid
# for data_idx in "${data_idxs[@]}"; do
#     $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=srafgen
#     mv tmp/grad_1024x1024 tmp/grad_1024x1024_srafgen$data_idx
# done
# $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=true logger.aim.experiment=srafgen



# data_idxs=(1)
res=mini
for data_idx in "${data_idxs[@]}"; do
    $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=true logger.aim.experiment=srafgen
    # mv tmp/grad_1024x1024 tmp/grad_1024x1024_srafgen$data_idx
done
