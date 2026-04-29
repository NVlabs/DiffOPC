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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from rich import print as rprint


def arr_bound(arr, name):
    rprint(f"\n=============[blue]{name}[/blue]================")
    # rprint(arr)
    min_wh = 7
    if arr.shape[0] >= min_wh and len(list(arr.shape)) > 1:
        lefti = arr.shape[0] // 2 - 4
        rprint(
            f"\n*************[blue]{name}[{lefti}:{lefti+min_wh},{lefti}:{lefti+min_wh}][/blue]*************"
        )
        rprint(arr[lefti : lefti + min_wh, lefti : lefti + min_wh])
        rprint(
            f"*************[blue]{name}[{lefti}:{lefti+7},{lefti}:{lefti+7}][/blue]*************\n"
        )
    else:
        rprint(arr)
    rprint(f"[blue]{name}[/blue].shape: {arr.shape}")
    rprint(f"[blue]{name}[/blue].dtype: {arr.dtype}")
    rprint(f"[blue]{name}[/blue] [red]sum[/red]: {np.sum(arr)}")
    if arr.dtype == "complex128":
        rprint(f"[blue]{name}.real[/blue] sum: {np.sum(arr.real)}")
        rprint(f"[blue]{name}.imag[/blue] sum: {np.sum(arr.imag)}")
        rprint(f"[blue]{name}.real[/blue] max: {np.max(arr.real)}")
        rprint(f"[blue]{name}.real[/blue] min: {np.min(arr.real)}")
        rprint(f"[blue]{name}.imag[/blue] max: {np.max(arr.imag)}")
        rprint(f"[blue]{name}.imag[/blue] min: {np.min(arr.imag)}")
    else:
        rprint(f"[blue]{name}[/blue] max: {np.max(arr)}")
        rprint(f"[blue]{name}[/blue] min: {np.min(arr)}")
    rprint(f"==============[blue]{name}[/blue]======================\n")


def torch_arr_bound(arr, name):
    rprint(f"\n=============[yellow]{name}[/yellow]================")
    # rprint(arr)
    # all the edges are Nx2x2
    N = arr.shape[0] // 2
    min_wh = 8
    if arr.shape[0] >= min_wh and len(list(arr.shape)) > 1:
        lefti = N - min_wh // 2
        rprint(f"\n*************[blue]{name}[{lefti}:{lefti+min_wh}][/blue]*************")
        rprint(arr[lefti : lefti + min_wh])
        rprint("****************************************************\n")
    else:
        rprint(arr)
    rprint(f"[blue]{name}[/blue].shape: {arr.shape}")
    rprint(f"[blue]{name}[/blue].dtype: {arr.dtype}")
    rprint(f"[blue]{name}[/blue] [red]sum[/red]: {torch.sum(arr)}")
    if arr.dtype == torch.complex128:
        rprint(f"[blue]{name}.real[/blue] sum: {torch.sum(arr.real)}")
        rprint(f"[blue]{name}.imag[/blue] sum: {torch.sum(arr.imag)}")
        rprint(f"[blue]{name}.real[/blue] max: {torch.max(arr.real)}")
        rprint(f"[blue]{name}.real[/blue] min: {torch.min(arr.real)}")
        rprint(f"[blue]{name}.imag[/blue] max: {torch.max(arr.imag)}")
        rprint(f"[blue]{name}.imag[/blue] min: {torch.min(arr.imag)}")
    else:
        rprint(f"[blue]{name}[/blue] max: {torch.max(arr)}")
        rprint(f"[blue]{name}[/blue] min: {torch.min(arr)}")
    rprint("====================================================\n")


def delta_np_torch(arr: np.array, tarr: torch.tensor):
    t_np_arr = torch.from_numpy(arr)
    rprint("=============[blue]The delta is : [/blue]=============")
    rprint(f"{torch.sum(t_np_arr - tarr)}\n")


def plot(ndarray):
    """Plots Transversed image, with origin (0,0) at the lower left corner."""

    plt.imshow(ndarray.T, origin="lower")
    plt.show()
