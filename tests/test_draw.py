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

import math
import os
import sys

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from test_data import edge_polygon_ids, edges, vertice_polygon_ids, vertices


def sort_polygon_vertices(vertices):
    # Calculate the centroid coordinates
    centroid_x = sum(v[0] for v in vertices) / len(vertices)
    centroid_y = sum(v[1] for v in vertices) / len(vertices)

    # Calculate the polar angle of each vertex relative to the centroid
    angles = []
    for v in vertices:
        dx = v[0] - centroid_x
        dy = v[1] - centroid_y
        angle = math.atan2(dy, dx)
        angles.append((angle, v))

    # Sort the vertices based on their polar angles
    sorted_vertices = sorted(
        angles, key=lambda x: (x[0], math.hypot(x[1][0] - centroid_x, x[1][1] - centroid_y))
    )

    # Extract the sorted vertex coordinates
    sorted_vertices = [v[1] for v in sorted_vertices]

    return sorted_vertices


def draw_filled_polygon(image, vertices):
    """Function to draw a filled polygon on a given image.

    Arguments:
    image -- A numpy array representing the image
    vertices -- A list of vertex coordinates in the format [(x1, y1), (x2, y2), ..., (xn, yn)]

    Returns:
    The modified image with the filled polygon
    """
    # Create a copy of the original image to avoid modifying it directly
    overlay = image.copy()

    # Convert the list of vertices to a numpy array
    vertices = np.array(vertices, dtype=np.int32)

    # Reshape the vertices array to have shape (num_vertices, 1, 2)
    vertices = vertices.reshape((-1, 1, 2))

    # Draw the filled polygon on the overlay image
    # cv2.fillPoly(overlay, [vertices], (255, 255, 255))
    cv2.drawContours(overlay, [vertices], -1, (255, 255, 255), -1)

    # Combine the original image with the overlay using bitwise OR operation
    result = cv2.bitwise_or(image, overlay)

    return result


def draw_image_by_vertices(vertices, vertice_polygon_ids, show_pid=0):
    # Create a blank image
    blank_image = np.zeros((2048, 2048, 3), dtype=np.int32)

    # Define the vertices of the polygon
    # vertices = [(100, 100), (500, 100), (500, 500), (100, 500)]
    unique_ids = torch.unique(vertice_polygon_ids)

    for idx in unique_ids:
        polygon_vertices = vertices[vertice_polygon_ids == idx]
        if show_pid is not None:
            if idx == show_pid:
                # Draw the filled polygon on the blank image
                # result_image = draw_filled_polygon(blank_image, sorted_vertices)
                blank_image = cv2.fillPoly(
                    blank_image, [polygon_vertices.numpy().astype(int)], (255, 255, 255)
                )
        else:
            blank_image = cv2.fillPoly(
                blank_image, [polygon_vertices.numpy().astype(int)], (255, 255, 255)
            )

    # Display the result
    # cv2.imshow("Filled Polygon", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(blank_image)
    plt.show()


def draw_edge_params(edge_params, edge_polygon_ids, shape, show_pid=0, show=True):
    image = torch.zeros(shape)
    edge_params_clone = edge_params.clone().detach()
    unique_ids = torch.unique(edge_polygon_ids)
    for idx in unique_ids:
        if idx == show_pid:
            # polygon_edges = edge_params_clone[edge_polygon_ids == idx]
            polygon_edge_indices = torch.where(edge_polygon_ids == idx)[0]
            polygon_edges = edge_params_clone[polygon_edge_indices]
            for edge in polygon_edges:
                start_point = edge[:, 0].clone().detach().int()
                end_point = edge[:, 1].clone().detach().int()

                if start_point[1] == end_point[1]:  # horizontal
                    if start_point[0] > end_point[0]:
                        start_point, end_point = end_point, start_point
                    image[start_point[1], start_point[0] : end_point[0] + 1] = 255
                else:
                    if start_point[1] > end_point[1]:  # vertical
                        start_point, end_point = end_point, start_point
                    image[start_point[1] : end_point[1] + 1, start_point[0]] = 255
    if show:
        plt.imshow(image.cpu().numpy())
        plt.show()
    return image


def draw_vertices(image, vertices, vertice_polygon_ids, shape, show_pid=0, show=True):
    if isinstance(image, torch.Tensor):
        image = image.clone().detach().cpu().numpy().astype(np.uint8)
    unique_ids = torch.unique(vertice_polygon_ids)
    for idx in unique_ids:
        if idx == show_pid:
            polygon_vertices = vertices[vertice_polygon_ids == idx]
            for vid, vertex in enumerate(polygon_vertices):
                vertex = vertex.clone().detach().int()
                x, y = int(vertex[0]), int(vertex[1])
                cv2.putText(image, f"{vid}", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

    if show:
        plt.imshow(image)
        plt.show()
    return image


def edges_to_vertices(edges, polygon_ids):
    """Convert polygon edge representation to vertice representation.

    Args:
        edges (torch.Tensor): Edge tensor of shape [N, 2, 2], where N is the number of edges,
                            2 represents the start and end points, and 2 represents 2-D coordinates (x, y)
        polygon_ids (torch.Tensor): Tensor of shape [N] containing polygon IDs for each edge
    Returns:
        vertices (torch.Tensor): Vertice tensor of shape [M, 2], where M is the total number of vertices,
                                and 2 represents 2-D coordinates (x, y)
        vertices_polygon_ids (torch.Tensor): Tensor of shape [M] containing polygon IDs for each vertex
    """
    # Get the unique polygon IDs
    unique_ids = torch.unique(polygon_ids)
    vertices_list = []
    vertices_polygon_ids_list = []

    # Iterate over each polygon ID
    for idx in unique_ids:
        # Get the indices of edges corresponding to the current polygon ID
        polygon_edge_indices = torch.where(polygon_ids == idx)[0]
        # Get the edges corresponding to the current polygon ID
        polygon_edges = edges[polygon_edge_indices]

        # Initialize the polygon vertices list
        polygon_vertices = []

        # Iterate over the edges and add unique vertices to the polygon vertices list
        for edge in polygon_edges:
            start_point = edge[:, 0]
            end_point = edge[:, 1]

            if len(polygon_vertices) == 0 or not torch.equal(start_point, polygon_vertices[-1]):
                polygon_vertices.append(start_point)

            if not torch.equal(end_point, polygon_vertices[-1]):
                polygon_vertices.append(end_point)

        # Convert the polygon vertices list to a tensor
        polygon_vertices = torch.stack(polygon_vertices)
        vertices_list.append(polygon_vertices)

        # Create polygon IDs for each vertex
        polygon_ids_for_vertices = torch.full(
            (polygon_vertices.shape[0],), idx, dtype=polygon_ids.dtype
        )
        vertices_polygon_ids_list.append(polygon_ids_for_vertices)

    # Concatenate the vertices and polygon IDs from all polygons
    vertices = torch.cat(vertices_list, dim=0)
    vertices_polygon_ids = torch.cat(vertices_polygon_ids_list, dim=0)

    return vertices, vertices_polygon_ids


if __name__ == "__main__":
    vertices = torch.tensor(vertices)
    edges = torch.tensor(edges)
    vertice_polygon_ids = torch.tensor(vertice_polygon_ids)
    edge_polygon_ids = torch.tensor(edge_polygon_ids)
    # draw_image_by_vertices(vertices, vertice_polygon_ids, show_pid=1)
    edge_image = draw_edge_params(edges, edge_polygon_ids, (2048, 2048, 3), show_pid=1, show=False)
    draw_vertices(
        edge_image, vertices, vertice_polygon_ids, (2048, 2048, 3), show_pid=1, show=True
    )

    new_vertices, new_vertice_polygon_ids = edges_to_vertices(edges, edge_polygon_ids)
    new_edge_image = draw_edge_params(
        edges, edge_polygon_ids, (2048, 2048, 3), show_pid=1, show=False
    )
    draw_vertices(
        new_edge_image,
        new_vertices,
        new_vertice_polygon_ids,
        (2048, 2048, 3),
        show_pid=1,
        show=True,
    )
    draw_image_by_vertices(new_vertices, new_vertice_polygon_ids, show_pid=None)
