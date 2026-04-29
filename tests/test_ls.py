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

import torch

params = torch.load("./tmp/LevelSet_test1.pt")
print(params.shape)


def find_zero_points(tensor, num_points=5, window_size=5):
    # Find the indices of all points with value 0
    zero_indices = torch.where((tensor <= 0) & (tensor >= -0.1))
    zero_indices = torch.stack(zero_indices, dim=1)

    # If the number of points with value 0 is less than num_points, take all the points
    if zero_indices.shape[0] < num_points:
        num_points = zero_indices.shape[0]

    # Randomly select num_points indices
    random_indices = torch.randperm(zero_indices.shape[0])[:num_points]
    selected_indices = zero_indices[random_indices]

    # List to store the results
    results = []

    # For each selected point, extract the surrounding 5x5 window data
    for idx in selected_indices:
        y, x = idx
        y_start = max(0, y - window_size // 2)
        y_end = min(tensor.shape[0], y + window_size // 2 + 1)
        x_start = max(0, x - window_size // 2)
        x_end = min(tensor.shape[1], x + window_size // 2 + 1)
        window_data = tensor[y_start:y_end, x_start:x_end]
        results.append((idx, window_data))

    return results


# Example usage
results = find_zero_points(params)

# Print the results
for idx, window_data in results:
    print(f"Zero point index: {idx}")
    print(f"Surrounding 5x5 window data:\n{window_data}\n")
