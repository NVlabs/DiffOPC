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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def find_intersection_and_adjust(line1, line2):
    """Adjusts two perpendicular line segments (one vertical and one horizontal) to meet exactly at
    their intersection point, ensuring that the start of the second line is the intersection point.
    Returns the adjusted line segments and the coordinates of the intersection point.

    Parameters:
    - line1: torch.tensor representing the first line segment.
    - line2: torch.tensor representing the second line segment.

    Both line1 and line2 are 2x2 tensors, where the first row contains x coordinates
    and the second row contains y coordinates of the line's start and end points, respectively.

    Returns:
    - Adjusted line segments as torch.tensors and the intersection point as a torch.tensor.
    """
    # Determine which line is vertical and which is horizontal
    is_line1_vertical = line1[0, 0] == line1[0, 1]

    # Extract vertical and horizontal lines
    vertical_line = line1 if is_line1_vertical else line2
    horizontal_line = line2 if is_line1_vertical else line1

    # Find intersection point
    intersection_x = vertical_line[0, 0]  # X-coordinate from the vertical line
    intersection_y = horizontal_line[1, 0]  # Y-coordinate from the horizontal line
    intersection_point = torch.tensor([intersection_x, intersection_y])

    # Adjust lines to meet at the intersection point
    if is_line1_vertical:
        # Adjust line1 (vertical) end point to intersection
        new_line1 = torch.tensor(
            [[vertical_line[0, 0], vertical_line[0, 0]], [vertical_line[1, 0], intersection_y]]
        )

        # Adjust line2 (horizontal) start point to intersection
        new_line2 = torch.tensor(
            [
                [intersection_x, horizontal_line[0, 1]],
                [horizontal_line[1, 1], horizontal_line[1, 1]],
            ]
        )
    else:
        # Adjust line1 (horizontal) end point to intersection
        new_line1 = torch.tensor(
            [
                [horizontal_line[0, 0], intersection_x],
                [horizontal_line[1, 0], horizontal_line[1, 0]],
            ]
        )

        # Adjust line2 (vertical) start point to intersection
        new_line2 = torch.tensor(
            [[vertical_line[0, 1], vertical_line[0, 1]], [intersection_y, vertical_line[1, 1]]]
        )

    return new_line1, new_line2, intersection_point


def draw_line_and_intersection(line1, line2, intersection=None):
    # Draw the line segments on the image
    img = np.zeros((1500, 1500, 3), dtype=np.uint8)
    cv2.line(
        img, tuple(line1[:, 0].int().tolist()), tuple(line1[:, 1].int().tolist()), (255, 0, 0), 2
    )
    cv2.line(
        img, tuple(line2[:, 0].int().tolist()), tuple(line2[:, 1].int().tolist()), (0, 255, 0), 2
    )
    # Draw the intersection point on the image
    if intersection is not None:
        cv2.circle(img, tuple(intersection.int().tolist()), 5, (0, 255, 255), -1)
    plt.imshow(img)
    plt.show()
    # return img


def print_line_and_intersection(line1, line2, intersection=None):
    if intersection is not None:
        line_type = "Adjusted"
    else:
        line_type = "Original"
    print(f"{line_type} Line 1: {line1} \n{line_type} Line 2: {line2}")


# group 1 : down left or down right


def lp0():
    line1 = torch.tensor([[680.0, 680.0], [800.0, 1134.0]])
    line2 = torch.tensor([[680.0, 1000.0], [1134.0, 1134.0]])
    return line1, line2


def lp1():
    line1 = torch.tensor([[680.0, 680.0], [800.0, 900.0]])
    line2 = torch.tensor([[680.0, 1000.0], [1134.0, 1134.0]])
    return line1, line2


def lp2():
    line1 = torch.tensor([[680.0, 680.0], [800.0, 900.0]])
    line2 = torch.tensor([[800.0, 1000.0], [1134.0, 1134.0]])
    return line1, line2


def lp3():
    line1 = torch.tensor([[680.0, 680.0], [800.0, 900.0]])
    line2 = torch.tensor([[600.0, 200.0], [1134.0, 1134.0]])
    return line1, line2


# group 2: up left or up right


def lp4():
    line1 = torch.tensor([[680.0, 680.0], [1200.0, 600.0]])
    line2 = torch.tensor([[980.0, 1300.0], [600.0, 600.0]])
    return line1, line2


def lp5():
    line1 = torch.tensor([[680.0, 680.0], [1200.0, 1000.0]])
    line2 = torch.tensor([[980.0, 1300.0], [600.0, 600.0]])
    return line1, line2


def lp6():
    line1 = torch.tensor([[680.0, 680.0], [1200.0, 1000.0]])
    line2 = torch.tensor([[480.0, 100.0], [600.0, 600.0]])
    return line1, line2


# group 3: right up or right down
def lp7():
    line1 = torch.tensor([[480.0, 100.0], [600.0, 600.0]])
    line2 = torch.tensor([[680.0, 680.0], [1200.0, 1000.0]])
    return line1, line2


# group 4 intersection at the middle
def lp8():
    line1 = torch.tensor([[680.0, 680.0], [800.0, 1134.0]])
    line2 = torch.tensor([[480.0, 1000.0], [1000.0, 1000.0]])
    return line1, line2


def test(line1, line2):
    # Example usage with the provided lines

    adjusted_line1, adjusted_line2, intersection = find_intersection_and_adjust(line1, line2)
    print(adjusted_line1[:, 1], adjusted_line2[:, 0], intersection)
    assert torch.equal(adjusted_line1[:, 1], adjusted_line2[:, 0])
    assert torch.equal(intersection, adjusted_line1[:, 1])
    draw_line_and_intersection(line1, line2)
    draw_line_and_intersection(adjusted_line1, adjusted_line2, intersection)
    print_line_and_intersection(line1, line2)
    print_line_and_intersection(adjusted_line1, adjusted_line2, intersection)


if __name__ == "__main__":
    # test(*lp0())
    # test(*lp1())
    # test(*lp2())
    # test(*lp3())
    # test(*lp4())
    # test(*lp5())
    # test(*lp6())
    # test(*lp7())
    test(*lp8())
