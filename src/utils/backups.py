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

# save edge params

# plt.show()

# fig, axs = plt.subplots(2, 4, figsize=(20, 12))
# if len(all_mask_edges) >= 8:
#     all_mask_edges = all_mask_edges[-8:]
# for i, ax in enumerate(axs.flat):
#     if i < len(all_mask_edges):
#         ax.imshow(all_mask_edges[i]["mask"])
#         ax.set_title(f"Iteration {all_masks[i]['iteration']}")
# plt.tight_layout()
# plt.savefig(f"{str(save_dir)}/EdgeILT_M1_test{case_id}_edge.png", dpi=300)

# for m in all_mask_edges:
#     plt.imsave(
#         f"{str(save_dir)}/EdgeILT_test{idx}_edge_{m['iteration']}.png",
#         m["mask"],
#         dpi=300,
#     )
# plt.show()
