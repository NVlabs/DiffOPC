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

import cv2
import numpy as np
from rich import print


class Design:
    """`Design` module for loading the GLP files."""

    def __init__(self, filename=None, down=1):
        self._filename = filename
        self._polygons = []
        if filename is None:
            return
        with open(filename) as fin:
            lines = fin.readlines()
        for line in lines:
            split = line.strip().split()
            if len(split) < 7:
                continue
            if split[0] == "RECT":
                info = split[3:7]
                frX = int(info[0])
                frY = int(info[1])
                toX = frX + int(info[2])
                toY = frY + int(info[3])
                coords = [
                    [frX // down, frY // down],
                    [frX // down, toY // down],
                    [toX // down, toY // down],
                    [toX // down, frY // down],
                ]
                self._polygons.append(coords)
            elif split[0] == "PGON":
                info = split[3:]
                coords = []
                for idx in range(0, len(info), 2):
                    coordX = int(info[idx])
                    coordY = int(info[idx + 1])
                    coords.append([coordX // down, coordY // down])
                self._polygons.append(coords)
        self._edges = []
        for polygon in self._polygons:
            edges = self.polygon_vertices_to_edges(polygon)
            self._edges.append(edges)

    @property
    def polygons(self):
        """Return all polygon representations."""
        return self._polygons

    @property
    def polygon_edges(self):
        return self._edges

    def polygon_vertices_to_edges(self, vertices):
        """Convert polygon vertices representation to edges representation.

        Args:
            vertices (list): List of polygon vertices in the format [[x1, y1], [x2, y2], ...].
        Returns:
            numpy.ndarray: Array of shape Nx2xD representing the edges of the polygon.
        """
        vertices = np.array(vertices)
        num_vertices = len(vertices)
        edges = np.zeros((num_vertices, 2, 2))  # Assuming 2D space (x, y)
        for i in range(num_vertices):
            edges[i, :, 0] = vertices[i]
            edges[i, :, 1] = vertices[(i + 1) % num_vertices]
        return edges

    def range(self):
        """Return the range for all polygons."""
        minX = 1e12
        minY = 1e12
        maxX = -1e12
        maxY = -1e12
        for polygon in self._polygons:
            for point in polygon:
                if point[0] < minX:
                    minX = point[0]
                if point[1] < minY:
                    minY = point[1]
                if point[0] > maxX:
                    maxX = point[0]
                if point[1] > maxY:
                    maxY = point[1]
        return minX, minY, maxX, maxY

    def move(self, deltaX, deltaY, offsetX=0, offsetY=0):
        """Move all polygon for a `deltaX` or `deltaY`"""
        for polygon in self._polygons:
            for point in polygon:
                point[0] += deltaX
                point[1] += deltaY

        for edges in self._edges:
            for edge in edges:
                edge[0, :] += deltaX + offsetX
                edge[1, :] += deltaY + offsetY

    def center(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512):
        """Move all polygon to center."""
        canvas = self.range()
        canvasX = canvas[2] - canvas[0]
        canvasY = canvas[3] - canvas[1]
        halfX = (sizeX - canvasX) // 2
        halfY = (sizeY - canvasY) // 2
        deltaX = halfX - canvas[0]
        deltaY = halfY - canvas[1]
        self.move(deltaX - offsetX, deltaY - offsetY, offsetX, offsetY)

    def image(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512, drawText=False):
        """Return the image representation for all images."""
        polygons = list(
            map(
                lambda x: np.array(x, np.int64) + np.array([[offsetX, offsetY]]),
                self._polygons,
            )
        )
        img = np.zeros([sizeX, sizeY], dtype=np.float32)
        for idx in range(len(polygons)):
            if drawText:
                for vert in polygons[idx]:
                    x, y = vert
                    cv2.putText(
                        img,
                        f"({x}, {y+20})",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
            cv2.fillPoly(img, [polygons[idx]], color=255)
        return img

    def image_edges(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512):
        """Return the image representation for all images."""
        edges = self.polygon_edges.copy()
        img = np.zeros([sizeX, sizeY], dtype=np.float32)
        for idx in range(len(edges)):
            # idx, how many polygons we have
            edges[idx][:, 0] += offsetX
            edges[idx][:, 1] += offsetY
            for edge in edges[idx]:
                # edge in each polygon
                # vector = edge[:, 1] - edge[:, 0]
                # length = np.linalg.norm(vector)
                # direction = vector / length
                # print(direction)
                # print(edge[:, 0].astype(int), edge[:, 1].astype(int))
                cv2.line(
                    img, tuple(edge[:, 0].astype(int)), tuple(edge[:, 1].astype(int)), color=255
                )
        return img

    def mat(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512):
        """Get mat."""
        return self.image(sizeX, sizeY, offsetX, offsetY) / 255.0

    def export(self, filename):
        """Export GLP files."""
        with open(filename, "w") as fout:
            fout.write("BEGIN     /* The metadata are invalid */\n")
            fout.write("EQUIV  1  1000  MICRON  +X,+Y\n")
            fout.write("CNAME Temp_Top\n")
            fout.write("LEVEL M1\n")
            fout.write("\n")
            fout.write("CELL Temp_Top PRIME\n")
            for kdx, polygon in enumerate(self._polygons):
                info = ""
                for point in polygon:
                    info += " " + str(point[0]) + " " + str(point[1])
                fout.write(f"   PGON N M1 {info}\n")
            fout.write("ENDMSG\n")

    def split(self, sizeX=16384, sizeY=16384, strideX=4096, strideY=4096, write=True):
        """Split polygon into different range."""
        minX, minY = 1e12, 1e12
        maxX, maxY = -1e12, -1e12
        ranges = []
        for polygon in self._polygons:
            minXpoly, minYpoly = 1e12, 1e12
            maxXpoly, maxYpoly = -1e12, -1e12
            for coord in polygon:
                if coord[0] > maxX:
                    maxX = coord[0]
                if coord[1] > maxY:
                    maxY = coord[1]
                if coord[0] < minX:
                    minX = coord[0]
                if coord[1] < minY:
                    minY = coord[1]
                if coord[0] > maxXpoly:
                    maxXpoly = coord[0]
                if coord[1] > maxYpoly:
                    maxYpoly = coord[1]
                if coord[0] < minXpoly:
                    minXpoly = coord[0]
                if coord[1] < minYpoly:
                    minYpoly = coord[1]
            ranges.append([minXpoly, minYpoly, maxXpoly, maxYpoly])
        print(f"[Design.split]: range ({minX, minY}) -> ({maxX, maxY})")

        intervalX = maxX - minX
        intervalY = maxY - minY
        stepsX = round((intervalX - (sizeX - strideX)) / strideX)
        stepsY = round((intervalY - (sizeY - strideY)) / strideY)
        print(f"[Design.split]: tiles ({stepsX, stepsY})")

        offsets = [[(None, None) for _ in range(stepsY)] for _ in range(stepsY)]
        visited = [False for _ in range(len(self._polygons))]

        for idx in range(stepsX):
            for jdx in range(stepsY):
                startX = minX + idx * strideX
                startY = minY + jdx * strideY
                endX = startX + sizeX
                endY = startY + sizeY
                polygons = []
                for kdx, polygon in enumerate(self._polygons):
                    # if ranges[kdx][0] >= endX or ranges[kdx][1] >= endY or ranges[kdx][2] < startX or ranges[kdx][3] < startY:
                    if (
                        ranges[kdx][0] >= startX
                        and ranges[kdx][1] >= startY
                        and ranges[kdx][2] < endX
                        and ranges[kdx][3] < endY
                    ):
                        polygons.append(polygon)
                        visited[kdx] = True
                offset = (startX, startY)
                if write:
                    filename = self._filename[:-4] + f"__{idx}_{jdx}" + ".glp"
                    with open(filename, "w") as fout:
                        fout.write("BEGIN     /* The metadata are invalid */\n")
                        fout.write("EQUIV  1  1000  MICRON  +X,+Y\n")
                        fout.write("CNAME Temp_Top\n")
                        fout.write("LEVEL M1\n")
                        fout.write("\n")
                        fout.write("CELL Temp_Top PRIME\n")
                        for polygon in polygons:
                            info = ""
                            for point in polygon:
                                info += (
                                    " "
                                    + str(point[0] - offset[0])
                                    + " "
                                    + str(point[1] - offset[1])
                                )
                            fout.write(f"   PGON N M1 {info}\n")
                        fout.write("ENDMSG\n")
        countCross = 0
        for kdx, polygon in enumerate(self._polygons):
            if not visited[kdx]:
                countCross += 1
        if write:
            filename = self._filename[:-4] + "__cross" + ".glp"
            with open(filename, "w") as fout:
                fout.write("BEGIN     /* The metadata are invalid */\n")
                fout.write("EQUIV  1  1000  MICRON  +X,+Y\n")
                fout.write("CNAME Temp_Top\n")
                fout.write("LEVEL M1\n")
                fout.write("\n")
                fout.write("CELL Temp_Top PRIME\n")
                for kdx, polygon in enumerate(self._polygons):
                    if not visited[kdx]:
                        info = ""
                        for point in polygon:
                            info += " " + str(point[0]) + " " + str(point[1])
                        fout.write(f"   PGON N M1 {info}\n")
                fout.write("ENDMSG\n")

        return countCross


if __name__ == "__main__":
    # design = Design("./benchmark/ICCAD2013/M1_test1.glp")
    # design.image()

    # design = Design("tmp/gcd.glp")
    # print(f"Range: {design.range()}")
    # size = design.range()
    # sizeILT = int(math.ceil(max(size)/2048) * 2048)
    # img = design.image(sizeX=sizeILT, sizeY=sizeILT, offsetX=0, offsetY=0)
    # cv2.imwrite("tmp/tmp.png", img)
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()

    # design = Design("work/gds_diff/gcd/gcd.glp")
    # design = Design("work/gds_diff/aes/aes_cipher_top.glp")
    # mask_file = "./benchmark/edge_bench/edge_test1.glp"
    mask_file = "./benchmark/edge_bench/M1_part_test1.glp"
    # mask shape y, x
    mask_shape = (512, 1280)
    mask = Design(mask_file)
    # mask_shape = (512, 512)
    # mask.center(mask_shape[0], mask_shape[1], 0, 0)

    img = mask.image(mask_shape[0], mask_shape[1], 100, 200)
    print(mask.polygons)
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()

    print(mask.polygon_edges)
    img_edge = mask.image_edges(mask_shape[0], mask_shape[1], 100, 200)
    plt.imshow(img_edge)
    plt.show()
    # count = design.split(sizeX=65536, sizeY=65536, strideX=16384, strideY=16384, write=True)
    # print(count)
