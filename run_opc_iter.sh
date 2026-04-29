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

device_id=0

steps=(1 2 4 8)
iters=(60 70 80 90)

# Exp 1
exp_name="iter_seg60"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=60 opc.Iterations=$iter opc.IsInsertSRAF=False opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done

exp_name="iter_seg60_sraf"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=60 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done


# Exp 2
exp_name="iter_seg80"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=80 opc.Iterations=$iter opc.IsInsertSRAF=False opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done

exp_name="iter_seg80_sraf"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=80 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done


# Exp 3
exp_name="iter_seg100"

for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=100 opc.Iterations=$iter opc.IsInsertSRAF=False opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done


exp_name="iter_seg100_sraf"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=100 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done
