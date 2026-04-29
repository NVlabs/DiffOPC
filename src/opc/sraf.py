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

from collections import deque

import numpy as np
import torch


def case_to_offset(square_case):
    top = 0, 1 / 2
    bottom = 1, 1 / 2
    left = 1 / 2, 0
    right = 1 / 2, 1

    if square_case == 1:
        return (top, left)
    elif square_case == 2:
        return (right, top)
    elif square_case == 3:
        return (right, left)
    elif square_case == 4:
        return (left, bottom)
    elif square_case == 5:
        return (top, bottom)
    elif square_case == 6:
        return (left, top)
    elif square_case == 7:
        return (right, bottom)
    elif square_case == 8:
        return (bottom, right)
    elif square_case == 9:
        return (top, right)
    elif square_case == 10:
        return (bottom, top)
    elif square_case == 11:
        return (bottom, left)
    elif square_case == 12:
        return (left, right)
    elif square_case == 13:
        return (top, right)
    elif square_case == 14:
        return (left, top)
    else:
        return (top, top)


def case_to_offset_degen(square_case):
    top = 0, 1 / 2
    bottom = 1, 1 / 2
    left = 1 / 2, 0
    right = 1 / 2, 1

    if square_case == 6:
        return (right, bottom)
    elif square_case == 9:
        return (bottom, left)
    else:
        return (top, top)


# modified from https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours_cy.pyx
def assemble_contours(segments):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        from_point = tuple(from_point.tolist())
        to_point = tuple(to_point.tolist())
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            if tail is head:
                head.append(to_point)
            else:
                if tail_num > head_num:
                    head.extend(tail)
                    contours.pop(tail_num, None)
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:
                    tail.extendleft(reversed(head))
                    starts.pop(head[0], None)
                    contours.pop(head_num, None)
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:
            tail.appendleft(from_point)
            starts[from_point] = (tail, tail_num)
        else:
            head.append(to_point)
            ends[to_point] = (head, head_num)

    return [contour for _, contour in sorted(contours.items())]


def marching_squares(x):
    offset_tensor = torch.tensor([case_to_offset(i) for i in range(0, 16)], device=x.device)

    degen_offset_tensor = torch.tensor([case_to_offset_degen(i) for i in range(0, 16)], device=x.device)
    with torch.no_grad():
        weight = torch.tensor([(1, 2), (4, 8)], dtype=x.dtype, requires_grad=False, device=x.device)[None, None, :, :]
        conv_out = torch.nn.functional.conv2d(
            x[None, None, :, :],
            weight=weight,
            stride=(1, 1),
            padding=1,
        ).squeeze()
        isedge = (conv_out > 0) & (conv_out < 15)
        vertices = isedge.nonzero()
        type_indices = conv_out[isedge]
        offsets = offset_tensor[type_indices.tolist()]
        edges = vertices[:, None, :] + offsets

        # handle degenerate cases
        degen = (type_indices == 6) | (type_indices == 9)
        degen_type_indices = type_indices[degen]
        degen_vertices = vertices[degen]
        degen_offsets = degen_offset_tensor[degen_type_indices.tolist()]
        degen_edges = degen_vertices[:, None, :] + degen_offsets
        all_edges = torch.cat([edges, degen_edges], dim=0)
        contours = assemble_contours(all_edges)
        return [torch.tensor(c) for c in contours]


def contour_area(contour):
    # Close the contour by making sure we also account for the last edge connecting the last and first points
    contour = torch.cat([contour, contour[:1]], axis=0)

    # Compute the x and y differences for each edge in the contour
    dx = contour[1:, 0] - contour[:-1, 0]
    dy = contour[1:, 1] - contour[:-1, 1]

    # Calculate the area using the shoelace formula
    area = torch.abs(torch.sum(dx * contour[:-1, 1] - dy * contour[:-1, 0]) / 2.0)
    return area.item()


def find_centroid(contour):
    # Calculate the sum of all x coordinates and the sum of all y coordinates
    sum_x = torch.sum(contour[:, 0])
    sum_y = torch.sum(contour[:, 1])
    # Calculate the centroid
    centroid_x = sum_x / contour.shape[0]
    centroid_y = sum_y / contour.shape[0]
    return centroid_x.int().item(), centroid_y.int().item()


def rectangle_to_polygon(c_x, c_y, sraf_w, sraf_h):
    # print(f"type c_x {type(c_x)}, c_x {c_x}")
    # print(f"type c_y {type(c_y)}, c_y {c_y}")
    # print(f"type sraf_w {type(sraf_w)}, sraf_w {sraf_w}")
    # Calculate half of the width and height for easy computation
    half_w = sraf_w // 2
    half_h = sraf_h // 2

    # Calculate the coordinates of the rectangle's four corners
    # Top-left corner
    top_left = (c_x - half_w, c_y - half_h)
    # Top-right corner
    top_right = (c_x + half_w, c_y - half_h)
    # Bottom-right corner
    bottom_right = (c_x + half_w, c_y + half_h)
    # Bottom-left corner
    bottom_left = (c_x - half_w, c_y + half_h)
    # Return the list of vertices in the order specified by OpenCV's coordinate system
    return [top_left, top_right, bottom_right, bottom_left]


def polygon_vertices_to_edges(vertices, device):
    """Convert polygon vertices representation to edges representation.

    Args:
        vertices (list): List of polygon vertices in the format [[x1, y1], [x2, y2], ...].
    Returns:
        numpy.ndarray: Array of shape Nx2xD representing the edges of the polygon.
    """
    # vertices = np.array(vertices)
    num_vertices = len(vertices)
    vertices = torch.tensor(vertices, device=device)
    edges = torch.zeros((num_vertices, 2, 2))  # Assuming 2D space (x, y)
    for i in range(num_vertices):
        edges[i, :, 0] = vertices[i]
        edges[i, :, 1] = vertices[(i + 1) % num_vertices]
    return edges


def check_sraf_in_boundaries(c_x, c_y, boundaries):
    (x_min, y_min, x_max, y_max) = boundaries
    if c_x < x_min or c_x > x_max:
        return False
    if c_y < y_min or c_y > y_max:
        return False
    return True


def get_sraf_polys(
    grad_image,
    min_contour_area=400,
    min_contour_wh_rule=20,
    initial_sraf_wh=60,
    boundaries=None,
):
    sraf_contours = marching_squares(grad_image)
    sraf_polys = []
    for contour in sraf_contours:
        area = contour_area(contour)
        if area > min_contour_area:
            contour_cartesian_coords = contour[:, [1, 0]]
            min_x = contour_cartesian_coords[:, 0].min().item()
            min_y = contour_cartesian_coords[:, 1].min().item()
            max_x = contour_cartesian_coords[:, 0].max().item()
            max_y = contour_cartesian_coords[:, 1].max().item()
            width = max_x - min_x
            height = max_y - min_y
            min_contour_wh = min(width, height)

            if min_contour_wh < min_contour_wh_rule:
                continue
            else:
                hw_ratio = height / width
                c_x, c_y = find_centroid(contour_cartesian_coords)
                if not check_sraf_in_boundaries(c_x, c_y, boundaries):
                    continue
                min_wh = initial_sraf_wh
                if width < min_wh or height < min_wh:
                    min_wh = min(width, height)
                if hw_ratio > 1:
                    sraf_w = min_wh
                    sraf_h = min_wh * hw_ratio
                else:
                    sraf_h = min_wh
                    sraf_w = min_wh / hw_ratio
                p_sraf = rectangle_to_polygon(c_x, c_y, sraf_w, sraf_h)
                p_sraf = np.array(p_sraf, dtype=np.int32)
                sraf_polys.append(p_sraf)
    return sraf_polys


def get_sraf_edges(
    mask,
    forbidden_mask,
    threshold_min=0.3,
    threshold_max=1,
    min_contour_area=400,
    min_contour_wh_rule=20,
    initial_sraf_wh=60,
    boundaries=None,
):
    assert boundaries is not None
    # (x_min, y_min, x_max, y_max) = boundaries
    binary_mask = mask.clone().detach()
    grad_map_clone = mask.grad.clone().detach()
    forbidden_mask = forbidden_mask.clone().detach()
    masked_grad_map = torch.zeros_like(grad_map_clone)
    masked_grad_map[binary_mask < 0.5] = grad_map_clone[binary_mask < 0.5]
    masked_grad_map[forbidden_mask > 0.5] = 0
    masked_grad_minus = torch.zeros_like(masked_grad_map)
    masked_grad_minus[masked_grad_map < 0] = masked_grad_map[masked_grad_map < 0]
    masked_grad_minus = -masked_grad_minus
    grad_image = torch.zeros_like(masked_grad_minus)
    masked_grad_max = masked_grad_minus.max()
    max_indices = (masked_grad_minus >= (threshold_min) * masked_grad_max) & (
        masked_grad_minus <= (threshold_max) * masked_grad_max
    )
    grad_image[max_indices] = 1
    sraf_contours = marching_squares(grad_image)
    sraf_edges = []
    for contour in sraf_contours:
        area = contour_area(contour)
        # print(f"area : {area}")
        if area > min_contour_area:
            contour_cartesian_coords = contour[:, [1, 0]]
            min_x = contour_cartesian_coords[:, 0].min().item()
            min_y = contour_cartesian_coords[:, 1].min().item()
            max_x = contour_cartesian_coords[:, 0].max().item()
            max_y = contour_cartesian_coords[:, 1].max().item()
            width = max_x - min_x
            height = max_y - min_y
            min_contour_wh = min(width, height)

            if min_contour_wh < min_contour_wh_rule:
                continue
            else:
                hw_ratio = height / width
                c_x, c_y = find_centroid(contour_cartesian_coords)
                # seg_length = 60
                if not check_sraf_in_boundaries(c_x, c_y, boundaries):
                    continue
                min_wh = initial_sraf_wh
                if width < min_wh or height < min_wh:
                    min_wh = min(width, height)
                if hw_ratio > 1:
                    sraf_w = min_wh
                    sraf_h = min_wh * hw_ratio
                else:
                    sraf_h = min_wh
                    sraf_w = min_wh / hw_ratio
                p_sraf = rectangle_to_polygon(c_x, c_y, sraf_w, sraf_h)
                # print(p_sraf)
                p_edges = polygon_vertices_to_edges(p_sraf, device=mask.device)
                # print(p_edges)
                sraf_edges.append(p_edges)
    return sraf_edges
