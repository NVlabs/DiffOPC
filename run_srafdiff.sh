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

seg_lengths=(60 80 100)

#sraf_low_opc_high

sraf_res=low
opc_res=high
aim_folder=aim_0419_sraf_${sraf_res}_opc_${opc_res}

# seg_length exp
for seg_length in ${seg_lengths[@]}; do
    exp_name=s_${sraf_res}_o_${opc_res}_seg_${seg_length}
    $python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res opc.high.SEG_LENGTH=$seg_length extras.print_config=false logger.aim.experiment=$exp_name logger.aim.repo=$aim_folder
done

# SRAF forbidden exp
sraf_forbiddens=(10 15 20 30)
for sraf_forbidden in ${sraf_forbiddens[@]}; do
    exp_name=s_${sraf_res}_o_${opc_res}_sforbidden_${sraf_forbidden}
    $python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res sraf.low.SRAF_FORBIDDEN=$sraf_forbidden extras.print_config=false logger.aim.experiment=$exp_name logger.aim.repo=$aim_folder
done

# SRAF initial wh exp
sraf_init_whs=(10 15 20)
for sraf_init_wh in ${sraf_init_whs[@]}; do
    exp_name=s_${sraf_res}_o_${opc_res}_swh_${sraf_init_wh}
    $python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res sraf.low.SRAF_initial_sraf_wh=$sraf_init_wh extras.print_config=false logger.aim.experiment=$exp_name logger.aim.repo=$aim_folder
done


# SRAF min threshold exp
sraf_thres_mins=(0.1 0.2 0.3 0.4)
for sraf_thres_min in ${sraf_thres_mins[@]}; do
    exp_name=s_${sraf_res}_o_${opc_res}_sthresmin_${sraf_thres_min}
    $python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res sraf.low.SRAF_threshold_min=$sraf_thres_min extras.print_config=false logger.aim.experiment=$exp_name logger.aim.repo=$aim_folder
done


# exp on max sraf candidates
max_sraf_grad_candidates=(1 2 3 4 5)
for sc in ${max_sraf_grad_candidates[@]}; do
    exp_name=s_${sraf_res}_o_${opc_res}_scandi_${sc}
    $python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res sraf.low.max_sraf_grad_candidates=$sc extras.print_config=false logger.aim.experiment=$exp_name logger.aim.repo=$aim_folder
done


# EPE parameters exp
epe_weights=(0 10 100 200 400 800)
for ew in ${epe_weights[@]}; do
    exp_name=s_${sraf_res}_o_${opc_res}_ew_${ew}
    $python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res opc.high.WeightEPE=$ew extras.print_config=false logger.aim.experiment=$exp_name logger.aim.repo=$aim_folder
done




#sraf_mid_opc_high
sraf_res=mid
opc_res=high
aim_folder=aim_0419_sraf_${sraf_res}_opc_${opc_res}
exp_name=s_${sraf_res}_o_${opc_res}
$python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res extras.print_config=false logger.aim.experiment=$exp_name logger.aim.repo=$aim_folder



#sraf_high_opc_high
sraf_res=high
opc_res=high
aim_folder=aim_0419_sraf_${sraf_res}_opc_${opc_res}
exp_name=s_${sraf_res}_o_${opc_res}
$python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res extras.print_config=false logger.aim.experiment=$exp_name logger.aim.repo=$aim_folder








# opc_res
# seg_length exp
# SRAF forbidden exp
# SRAF initial wh exp
# SRAF min threshold exp
# exp on max sraf candidates
# EPE loss exp
# EPE parameters exp
