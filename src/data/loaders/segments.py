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

import sys

sys.path.append(".")
import math
import multiprocessing as mp

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

# import pycommon.utils as common
import src.data.loaders.glp_seg as glp_seg
from src.data.datatype import COMPLEXTYPE, REALTYPE
from src.litho.simple import LithoSim
from src.opc.utils import (
    right_perpendicular_unit_vector,
    segment_polygon_edges_with_labels,
)


class Initializer:
    def __init__(self):
        pass

    def run(self, design, sizeX, sizeY, offsetX, offsetY, device, dtype=REALTYPE):
        pass


def segs2metadata(seg_params, start_polygon_id, device):
    edge_params = []
    polygon_ids = []
    direction_vectors = []
    velocities = []
    corner_ids = []
    epe_points = []
    seg_types = []
    for idx, poly in enumerate(seg_params):
        polygon_ids.append(torch.full((len(poly),), idx + start_polygon_id, dtype=torch.int32))
        is_clockwise = True if poly[0]["direction"][0] == 1 else False
        for seg in poly:
            edge_params.append(seg["segment"].detach().clone())
            direction_vectors.append(seg["direction"].detach().clone())
            # proceed to mark corners
            corner = 1 if "C" in seg["type"] else 0
            corner_ids.append(corner)
            # proceed to calculate the velocity
            velocity = right_perpendicular_unit_vector(seg["direction"])
            if not is_clockwise:
                velocity = -velocity
            velocity = torch.stack([velocity, velocity], dim=0)
            velocity = torch.transpose(velocity, 0, 1)
            velocities.append(velocity.round())
            epe_points.append(seg["epe_point"].detach().clone())
            seg_types.append(seg["type"])
    edge_params = torch.stack(edge_params, dim=0).to(device).requires_grad_(True)
    polygon_ids = torch.cat(polygon_ids, dim=0).to(device)
    direction_vectors = torch.stack(direction_vectors, dim=0).to(device)
    velocities = torch.stack(velocities, dim=0).to(device)
    corner_ids = torch.tensor(corner_ids, dtype=torch.int32, device=device)
    epe_points = torch.stack(epe_points, dim=0).to(device)
    return edge_params, polygon_ids, direction_vectors, velocities, corner_ids, epe_points, seg_types


class SegmentsInitTorch(Initializer):
    def __init__(self):
        super().__init__()

    def run(
        self,
        design,
        sizeX,
        sizeY,
        offsetX,
        offsetY,
        seg_length,
        sraf_forbidden,
        device,
        dtype=REALTYPE,
        start_segment_id=0,
        start_polygon_id=0,
    ):
        self.device = device
        design.center(sizeX, sizeY, offsetX, offsetY)
        target = torch.tensor(design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device)
        target_edges = design.polygon_edges
        seg_params = segment_polygon_edges_with_labels(target_edges, seg_length, start_segment_id=start_segment_id)

        edge_params, polygon_ids, direction_vectors, velocities, corner_ids, epe_points, seg_types = segs2metadata(
            seg_params, start_polygon_id=start_polygon_id, device=self.device
        )
        shape = (sizeX, sizeY)
        assert polygon_ids.shape[0] == edge_params.shape[0]
        assert polygon_ids.shape[0] == direction_vectors.shape[0]
        assert edge_params.shape == velocities.shape
        metadata = {
            "img_shape": shape,
            "polygon_ids": polygon_ids,
            "direction_vectors": direction_vectors,
            "velocities": velocities,
            "corner_ids": corner_ids,
            "sraf_forbidden": sraf_forbidden,
            "epe_points": epe_points,
            "seg_types": seg_types,
        }
        # print(edge_params[0:20])
        # print(epe_points[0:20])
        # print(direction_vectors[0:5])
        # print(velocities[0:5])
        # print(corner_ids[0:5])
        return target, edge_params, metadata


if __name__ == "__main__":
    import levelset as ilt

    maskfile = "./benchmark/edge_bench/edge_test1.glp"
    mask_shape = (512, 512)
    mref = glp_seg.Design(maskfile, down=1)
