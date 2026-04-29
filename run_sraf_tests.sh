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

# device_id=1
thres_mins=(0.1 0.2 0.3 0.4)
exp_name="thres_min"
for thres_min in "${thres_mins[@]}"; do
    $python src/diffopc.py opc.SRAF_threshold_min=$thres_min logger.aim.experiment=$exp_name
done



forbiddens=(10 20 40 60)
exp_name="sraf_forbidden"
for forbidden in "${forbiddens[@]}"; do
    $python src/diffopc.py opc.SRAF_FORBIDDEN=$forbidden logger.aim.experiment=$exp_name
done


areas=(300 400 600 900)
exp_name="sraf_area"
for area in "${areas[@]}"; do
    $python src/diffopc.py opc.SRAF_contour_area=$area logger.aim.experiment=$exp_name
done

initial_sraf_whs=(20 40 60 80)
exp_name="init_sraf_wh"
for wh in "${initial_sraf_whs[@]}"; do
    $python src/diffopc.py opc.SRAF_initial_sraf_wh=$wh logger.aim.experiment=$exp_name
done
