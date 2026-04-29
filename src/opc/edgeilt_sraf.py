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

import os
import sys

sys.path.append(".")
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from rich import print

import src.data.loaders.glp_seg as glp_seg
import src.data.loaders.segments as SegLoader
import src.opc.evaluation as evaluation
from src.data.datatype import REALTYPE
from src.data.loaders.segments import segs2metadata
from src.litho.simple import LithoSim
from src.opc.evaluation import EPE_CONSTRAINT
from src.opc.sraf import get_sraf_edges
from src.opc.utils import (
    adjust_corner_edge_params,
    edge_params2forbidden,
    edge_params_merge2mask,
    segment_polygon_edges_with_labels,
)
from src.utils.utils import yaml2Cfg

# import pylitho.exact as lithosim
# draw_edge_params,
# draw_grad_map,


class EdgeILTCfg:
    def __init__(self, config):
        # Read the config from file or a given dict
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, str):
            self._config = yaml2Cfg(config)
        elif isinstance(config, DictConfig):
            self._config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

        required = [
            "Iterations",
            "TargetDensity",
            "SigmoidSteepness",
            "WeightEPE",
            "WeightPVBL2",
            "WeightPVBand",
            "StepSize",
            "TileSizeX",
            "TileSizeY",
            "OffsetX",
            "OffsetY",
            "ILTSizeX",
            "ILTSizeY",
            "DownScale",
        ]
        for key in required:
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY", "DownScale"]
        for key in intfields:
            self._config[key] = int(self._config[key])
        floatfields = [
            "TargetDensity",
            "SigmoidSteepness",
            "WeightEPE",
            "WeightPVBL2",
            "WeightPVBand",
            "StepSize",
        ]
        for key in floatfields:
            self._config[key] = float(self._config[key])

    def __getitem__(self, key):
        return self._config[key]


def get_avg_grad_line(edge, grad_output):
    start_point = edge[:, 0].int()
    end_point = edge[:, 1].int()
    if start_point[1] == end_point[1]:  # horizontal
        if start_point[0] > end_point[0]:
            start_point, end_point = end_point, start_point
        selected_line = grad_output[start_point[1], start_point[0] : end_point[0] + 1]
    else:
        if start_point[1] > end_point[1]:  # vertical
            start_point, end_point = end_point, start_point
        selected_line = grad_output[start_point[1] : end_point[1] + 1, start_point[0]]
    average_value = selected_line.mean()
    return average_value


def get_avg_grad_points(edge, grad_output):
    start_point = edge[:, 0].int()
    end_point = edge[:, 1].int()
    grad_start = grad_output[start_point[1], start_point[0]]
    grad_end = grad_output[end_point[1], end_point[0]]
    average_value = (grad_start + grad_end) / 2
    return average_value


def get_avg_grad(edge, grad_output):
    mid_point = (edge[:, 0] + edge[:, 1]) / 2
    average_value = grad_output[mid_point[1].int(), mid_point[0].int()]
    return average_value


def save_masks(all_masks, save_dir, case_id):
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 4, figsize=(20, 12))
    if len(all_masks) >= 8:
        all_masks = all_masks[-8:]
    for i, ax in enumerate(axs.flat):
        if i < len(all_masks):
            ax.imshow(all_masks[i]["mask"])
            ax.set_title(f"Iteration {all_masks[i]['iteration']}")
    plt.tight_layout()
    plt.savefig(f"{str(save_dir)}/EdgeILT_M1_test{case_id}_mask.png", dpi=300)

    for m in all_masks:
        plt.imsave(
            f"{str(save_dir)}/EdgeILT_test{case_id}_mask_{m['iteration']}.png",
            m["mask"],
            dpi=300,
        )


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params):
        quantized = torch.round(params)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_params, metadata, iter_idx):
        # begin = time.time()
        ctx.metadata_param = metadata
        ctx.iter_idx = iter_idx
        binary_mask = edge_params_merge2mask(edge_params, metadata)
        ctx.save_for_backward(edge_params)
        # print(f"Binarize forward time: {time.time() - begin:.2f}s")
        return binary_mask

    @staticmethod
    def backward(ctx, grad_output):
        # begin = time.time()
        (edge_params,) = ctx.saved_tensors
        metadata = ctx.metadata_param
        velocities = metadata["velocities"]
        average_values = []
        for edge in edge_params:
            average_value = get_avg_grad(edge, grad_output)
            average_values.append(average_value)
        average_values = torch.stack(average_values)
        average_values = average_values.view(-1, 1, 1)
        grad_edge_params = average_values * velocities
        # print(f"Binarize backward time: {time.time() - begin:.2f}s")
        return grad_edge_params, None, None


class EdgeMerger(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_params, metadata):
        # begin = time.time()
        edge_params = adjust_corner_edge_params(edge_params, metadata)
        # print(f"EdgeMerger time: {time.time() - begin:.2f}s")
        return edge_params

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class EdgeILT(nn.Module):
    def __init__(self, lithosim):
        super().__init__()
        self._lithosim = lithosim

    def forward(self, edge_params, metadata, iter_idx):
        edge_params = StraightThroughEstimator.apply(edge_params)
        edge_params = EdgeMerger.apply(edge_params, metadata)
        mask = Binarize.apply(edge_params, metadata, iter_idx)
        mask.retain_grad()
        printedNom, printedMax, printedMin = self._lithosim(mask)
        return mask, printedNom, printedMax, printedMin, edge_params


class EdgeILTSolver:
    def __init__(
        self,
        config: EdgeILTCfg,
        lithosim: LithoSim,
        device: torch.device,
    ):
        super().__init__()
        self._config = config
        self._device = device
        self._edgeILT = EdgeILT(lithosim).to(self._device)
        self._filter = torch.zeros(
            [self._config["TileSizeX"], self._config["TileSizeY"]],
            dtype=REALTYPE,
            device=self._device,
        )
        self._filter[
            self._config["OffsetX"] : self._config["OffsetX"] + self._config["ILTSizeX"],
            self._config["OffsetY"] : self._config["OffsetY"] + self._config["ILTSizeY"],
        ] = 1

        self.grad_queue = [-1e5] * 5

    def trigger_insert_sraf(self, mask):
        if mask.grad is not None:
            min_grad = mask.grad.min().abs()
            # print(f"min: {min_grad}")
            self.grad_queue.append(min_grad)
            if len(self.grad_queue) > 5:
                self.grad_queue.pop(0)
            queue_max = max(self.grad_queue)
            print(f"queue_max: {queue_max}")
            if queue_max < 0.6:
                return True
            else:
                return False

    def init_sraf_params(self, mask, edge_params, metadata):
        edge_params = edge_params.clone().detach()
        forbidden_mask, _ = edge_params2forbidden(edge_params, metadata)
        sraf_threshold_min = self._config["SRAF_threshold_min"]
        x_min = self._config["OffsetX"] - self._config["SEG_LENGTH"]
        y_min = self._config["OffsetY"] - self._config["SEG_LENGTH"]
        x_max = self._config["OffsetX"] + self._config["ILTSizeX"] + self._config["SEG_LENGTH"]
        y_max = self._config["OffsetY"] + self._config["ILTSizeY"] + self._config["SEG_LENGTH"]
        sraf_edges = get_sraf_edges(
            mask,
            forbidden_mask,
            threshold_min=sraf_threshold_min,
            threshold_max=1,
            min_contour_area=self._config["SRAF_contour_area"],
            min_contour_wh_rule=self._config["SRAF_min_contour_wh_rule"],
            initial_sraf_wh=self._config["SRAF_initial_sraf_wh"],
            boundaries=(x_min, y_min, x_max, y_max),
        )
        assert len(sraf_edges) > 0, "No SRAF edges found!"
        start_polygon_id = metadata["polygon_ids"][-1].item() + 1
        start_segment_id = edge_params.shape[0] + 1
        sraf_seg_params = segment_polygon_edges_with_labels(sraf_edges, self._config["SEG_LENGTH"], start_segment_id)

        (
            sraf_edge_params,
            sraf_polygon_ids,
            sraf_direction_vectors,
            sraf_velocities,
            sraf_corner_ids,
        ) = segs2metadata(sraf_seg_params, start_polygon_id=start_polygon_id, device=self._device)
        new_edge_params = torch.cat([edge_params, sraf_edge_params], dim=0)
        metadata["polygon_ids"] = torch.cat([metadata["polygon_ids"], sraf_polygon_ids], dim=0)
        metadata["direction_vectors"] = torch.cat([metadata["direction_vectors"], sraf_direction_vectors], dim=0)
        metadata["velocities"] = torch.cat([metadata["velocities"], sraf_velocities], dim=0)
        metadata["corner_ids"] = torch.cat([metadata["corner_ids"], sraf_corner_ids], dim=0)
        return new_edge_params, metadata

    def cal_loss(self, target, printedNom, printedMax, printedMin, kernelCurv=None):
        l2loss = func.mse_loss(printedNom, target, reduction="sum")
        pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
        pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
        pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
        if kernelCurv is not None:
            curvature = func.conv2d(printedNom[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
            losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
            loss = (
                self._config["WeightL2"] * l2loss
                + self._config["WeightPVBL2"] * pvbl2
                + self._config["WeightPVBand"] * pvbloss
                + 2e2 * losscurv
            )
        else:
            loss = (
                self._config["WeightL2"] * l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss
            )
        return loss, l2loss, pvband, pvbl2, pvbloss

    def cal_epe_loss(self, target, printedNom, metadata):
        D = torch.square(target - printedNom)
        epe_points = metadata["epe_points"]
        seg_types = metadata["seg_types"]
        epe_loss = 0
        epe_range = 15
        for idx, epe_point in enumerate(epe_points):
            x = epe_point[0].int()
            y = epe_point[1].int()
            if seg_types[idx] == "H":
                D2 = D[y - epe_range : y + epe_range, x]
                D2 = torch.sum(D2)
                D2 = torch.sigmoid(D2 * self._config["SigmoidSteepness"])
            elif seg_types[idx] == "V":
                D2 = D[y, x - epe_range : x + epe_range]
                D2 = torch.sum(D2)
                D2 = torch.sigmoid(D2 * self._config["SigmoidSteepness"])
            else:
                # raise ValueError(f"Unknown segment type: {seg_types[idx]}")
                # ignore the corner segment
                continue
            epe_loss += D2
        return epe_loss

    def solve(self, target, edge_params, metadata, case_id=1, verbose=0):
        # Initialize

        # Optimizer
        if self._config["OPT"] == "sgd":
            opt = optim.SGD([edge_params], lr=self._config["StepSize"])
        else:
            opt = optim.Adam([edge_params], lr=self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        bestMaskIter = None
        all_masks = []
        # all_mask_edges = []

        # debug
        new_edge_params = None
        opc_idx = 0
        kernelCurv = torch.tensor(
            [[-1.0 / 16, 5.0 / 16, -1.0 / 16], [5.0 / 16, -1.0, 5.0 / 16], [-1.0 / 16, 5.0 / 16, -1.0 / 16]],
            dtype=REALTYPE,
            device=edge_params.device,
        )
        # kernelCurv = None

        # =========================================================================
        # OPC loop
        # =========================================================================
        for idx in range(self._config["Iterations"]):
            mask, printedNom, printedMax, printedMin, edge_params_clone = self._edgeILT(edge_params, metadata, idx)
            loss, l2loss, pvband, _, _ = self.cal_loss(target, printedNom, printedMax, printedMin, kernelCurv=kernelCurv)
            if self._config["EPELoss"]:
                epe_loss = self.cal_epe_loss(target, printedNom, metadata) * self._config["WeightEPE"]
                loss += epe_loss

            if self._config["VISUAL_DEBUG"]:
                if idx % 40 == 0:
                    mask_cpu = mask.clone().detach().cpu().numpy()
                    all_masks.append({"mask": mask_cpu, "iteration": idx})

            if not self._config["IsInsertSRAF"]:
                if bestParams is None or bestMask is None or loss < lossMin:
                    lossMin, l2Min, pvbMin = loss, l2loss, pvband
                    bestParams = edge_params_clone
                    bestMask = mask.detach().clone()
                    bestMaskIter = idx

            if verbose:
                print(f"[OPC Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            opt.zero_grad()
            loss.backward()
            opt.step()
            opc_idx = idx

            if self._config["IsInsertSRAF"]:
                # if self.trigger_insert_sraf(mask) or idx >= 70:
                if idx >= self._config["Iterations"] - 1:
                    if self._config["VISUAL_DEBUG"]:
                        mask_cpu = mask.clone().detach().cpu().numpy()
                        all_masks.append({"mask": mask_cpu, "iteration": idx})
                    # print("Insert SRAF!")
                    seg_length = self._config["SEG_LENGTH"]
                    new_edge_params, new_metadata = self.init_sraf_params(mask, edge_params, metadata, seg_length)
                    opc_idx = idx
                    # print(f"Insert SRAF at iteration {opc_idx}")
                    break

        # SRAF loop
        if self._config["IsInsertSRAF"] and (new_edge_params is not None):
            new_edge_params = new_edge_params.detach().clone().requires_grad_(True)
            if self._config["OPT"] == "sgd":
                opt = optim.SGD([new_edge_params], lr=self._config["StepSize"])
            else:
                opt = optim.Adam([new_edge_params], lr=self._config["StepSize"])

            for idx in range(self._config["SRAF_ITERATIONS"]):
                mask, printedNom, printedMax, printedMin, new_edge_params_clone = self._edgeILT(
                    new_edge_params, new_metadata, idx
                )
                loss, l2loss, pvband, _, _ = self.cal_loss(target, printedNom, printedMax, printedMin, kernelCurv=kernelCurv)

                if self._config["VISUAL_DEBUG"]:
                    if idx % 20 == 0:
                        mask_cpu = mask.clone().detach().cpu().numpy()
                        all_masks.append({"mask": mask_cpu, "iteration": opc_idx + idx})

                if verbose:
                    print(f"[OPC: {opc_idx}, SRAF Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

                if bestParams is None or bestMask is None or loss < lossMin:
                    lossMin, l2Min, pvbMin = loss, l2loss, pvband
                    bestParams = new_edge_params_clone
                    bestMask = mask.detach().clone()
                    bestMaskIter = opc_idx + idx

                opt.zero_grad()
                loss.backward()
                opt.step()

        # Visual Debug
        if self._config["VISUAL_DEBUG"]:
            report_dir = "report_sraf" if self._config["IsInsertSRAF"] else "report"
            if self._config["DownScale"] != 1:
                down_dir = f"down{self._config['DownScale']}x"
                save_dir = f"./tmp/{report_dir}/{down_dir}/M1_test{case_id}"
            else:
                save_dir = f"./tmp/{report_dir}/M1_test{case_id}"
            all_masks.append({"mask": bestMask.cpu().numpy(), "iteration": bestMaskIter})
            print(f"Saving masks to {save_dir}")
            save_masks(all_masks, save_dir, case_id)

        return l2Min, pvbMin, bestParams, bestMask, bestMaskIter


def serial():
    from omegaconf import OmegaConf

    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    targetsAll = []
    paramsAll = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    edgeILTCfg = OmegaConf.load("./configs/opc/direct_run.yaml")
    edgeILTCfg = OmegaConf.to_container(edgeILTCfg, resolve=True, throw_on_missing=True)
    cfg = EdgeILTCfg(edgeILTCfg)

    lithoCfg = OmegaConf.load("./configs/litho/default.yaml")
    litho = LithoSim(lithoCfg.litho_config, device)
    solver = EdgeILTSolver(cfg, litho, device)
    for idx in range(1, 11):
        design = glp_seg.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        # design = glp_seg.Design(f"./benchmark/edge_bench/edge_test{idx}.glp", down=SCALE)
        # design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, edge_params, metadata = SegLoader.SegmentsInitTorch().run(
            design,
            cfg["TileSizeX"],
            cfg["TileSizeY"],
            cfg["OffsetX"],
            cfg["OffsetY"],
            cfg["SEG_LENGTH"],
            cfg["SRAF_FORBIDDEN"],
        )
        begin = time.time()
        l2, pvb, bestMask, bestMaskIter = solver.solve(
            target,
            edge_params,
            metadata,
            case_id=idx,
            curv=None,
            verbose=cfg["VERBOSE"],
        )
        runtime = time.time() - begin
        if cfg["Evaluation"]:
            l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, device=device, scale=SCALE, shots=True)

            cv2.imwrite(f"./tmp/EdgeILT_test{idx}.png", (bestMask * 255).detach().cpu().numpy())

            print(
                f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; BestIter: {bestMaskIter} SolveTime: {runtime:.2f}s"
            )

            l2s.append(l2)
            pvbs.append(pvb)
            epes.append(epe)
            shots.append(shot)
            runtimes.append(runtime)
        else:
            print(f"Testcase {idx} finished.")
    if cfg["Evaluation"]:
        print(
            f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime: {np.mean(runtimes):.2f}s"
        )


if __name__ == "__main__":
    serial()
    # parallel()
