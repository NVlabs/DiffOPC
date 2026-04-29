import sys

sys.path.append(".")
import argparse
import logging
import math

# import pylitho.exact as lithosim
import os
import time

import cv2
import numpy as np
import pycommon.glp as glp
import pycommon.utils as common
import pyilt.evaluation as evaluation
import pyilt.initializer as initializer
import pylitho.simple as lithosim
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from PIL import Image
from pycommon.settings import *
from visualization import (
    coord2circle,
    vis_grad_pixel,
    vis_params_circle,
    vis_params_level,
    vis_params_pixel,
    vis_radius_level,
)

torch.autograd.set_detect_anomaly(True)

logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.DEBUG)
ALPHA = 10
LR_COORD = 0.1
MARGIN = 356
MIN_RADIUS = 3
MAX_RADIUS = 19


class CircleCfg:
    def __init__(self, config):
        # Read the config from file or a given dict
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, str):
            self._config = common.parseConfig(config)

        required = [
            "Iterations",
            "TargetDensity",
            "SigmoidSteepness",
            "SigmoidOffset",
            "WeightPVBand",
            "WeightNom",
            "WeightMin",
            "WeightMax",
            "StepSize",
            "WeightCurv",
            "WeightArea",
            "ScaleTanh",
            "TileSizeX",
            "TileSizeY",
            "OffsetX",
            "OffsetY",
            "ILTSizeX",
            "ILTSizeY",
            "ThreshArea",
            "ThreshRange",
        ]
        for key in required:
            assert key in self._config, f"[PixelILT]: Cannot find the config {key}."
        intfields = [
            "Iterations",
            "TileSizeX",
            "TileSizeY",
            "OffsetX",
            "OffsetY",
            "ILTSizeX",
            "ILTSizeY",
            "ThreshArea",
            "ThreshRange",
            "LowResFactor",
        ]
        for key in intfields:
            self._config[key] = int(self._config[key])
        floatfields = [
            "TargetDensity",
            "SigmoidSteepness",
            "SigmoidOffset",
            "WeightPVBand",
            "WeightNom",
            "WeightMin",
            "WeightMax",
            "StepSize",
            "WeightCurv",
            "WeightArea",
            "ScaleTanh",
        ]
        for key in floatfields:
            self._config[key] = float(self._config[key])

    def __getitem__(self, key):
        return self._config[key]


def get_borders(H, W, i, j, window_size, down_factor=1):
    left = max(0, j - (window_size - 1) // (2 * down_factor))
    right = min(W, 1 + j + (window_size - 1) // (2 * down_factor))
    up = max(0, i - (window_size - 1) // (2 * down_factor))
    down = min(H, 1 + i + (window_size - 1) // (2 * down_factor))
    return int(left), int(right), int(up), int(down)


# eq 11
def circle2pixel_with_coord(
    x_coord,
    y_coord,
    radius,
    value,
    window_size=39,
    params_pixel_shape=(2048, 2048),
    no_quant=False,
):
    point_num = x_coord.shape[0]
    H, W = params_pixel_shape
    params_pixel = torch.zeros(params_pixel_shape, device=x_coord.device, dtype=value.dtype)
    all_time = []
    for i in range(point_num):
        start = time.time()
        x, y, r, v = x_coord[i], y_coord[i], radius[i], value[i]
        left, right, up, down = get_borders(H, W, y.item(), x.item(), window_size)
        size_X, size_Y = right - left, down - up
        window_x_coord = torch.arange(left, right, 1).repeat(size_Y, 1).to(DEVICE)
        window_y_coord = torch.arange(up, down, 1).reshape(size_Y, -1).repeat(1, size_X).to(DEVICE)
        dist = torch.pow(
            torch.pow(window_x_coord - x, 2) + torch.pow(window_y_coord - y, 2) + 1e-8, 0.5
        )
        activation = torch.sigmoid(ALPHA * (r - dist)) * v
        params_pixel_region = params_pixel[up:down, left:right]
        params_pixel_region = torch.cat(
            (params_pixel_region.unsqueeze(0), activation.unsqueeze(0)), dim=0
        )
        params_pixel[up:down, left:right] = torch.max(params_pixel_region, dim=0)[0]
        end = time.time()
        all_time.append(end - start)
    return params_pixel


def get_full_mask(
    params_circle,
    scale,
    params_pixel=None,
    mode="scale",
    value_th=0.5,
    rec=True,
    with_coord=False,
    radius_clipping=True,
    min_radius=MIN_RADIUS,
    max_radius=MAX_RADIUS,
):
    if isinstance(params_circle, tuple):
        if with_coord:
            params_circle = coord2circle(
                params_circle[0],
                params_circle[1],
                params_circle[2],
                params_circle[3],
                shape=(2048 // scale, 2048 // scale, 2),
            )
            params_radius, params_value = params_circle[:, :, 0], params_circle[:, :, 1]
        else:
            params_radius, params_value = params_circle[0], params_circle[1]
    else:
        params_radius, params_value = params_circle[:, :, 0], params_circle[:, :, 1]
    assert isinstance(params_radius, torch.Tensor)
    params_circle_radius = params_radius.clone().detach()
    params_circle_radius = params_circle_radius.cpu().numpy()
    params_circle_value = params_value.clone().detach().cpu().numpy()

    y, x = (params_circle_value > value_th).nonzero()
    points = [
        (x[i], y[i], params_circle_radius[y[i], x[i]], params_circle_value[y[i], x[i]])
        for i in range(len(x))
    ]
    shot = len(points)
    H, W = params_radius.shape
    full_mask = np.zeros((H * scale, W * scale))
    full_params_circle = np.zeros((H * scale, W * scale, 2))
    for p in points:
        r, v = p[2], p[3]
        if rec:
            r = r + math.log(2 * v - 1) / ALPHA
        r *= scale
        if r > max_radius * scale:
            r = max_radius * scale
        if r < min_radius * scale:
            if r > (min_radius * 0.5) * scale:
                r = min_radius * scale
            else:
                continue
        x, y = p[0] * scale, p[1] * scale
        cv2.circle(full_mask, (x, y), round(r), 255, -1)
        full_params_circle[y, x, 0], full_params_circle[y, x, 1] = round(r), p[3]

    full_mask = full_mask / 255.0

    return full_mask, full_params_circle, shot


def eval_simple(params_pixel, target, litho_model, value_th=0.5):
    if isinstance(params_pixel, np.ndarray):
        params_pixel = torch.tensor(params_pixel, dtype=target.dtype, device=target.device)
    mask_bin = params_pixel.clone().detach()
    mask_bin[mask_bin <= value_th] = 0.0
    mask_bin[mask_bin > value_th] = 1.0
    printedNom_bin, printedMax_bin, printedMin_bin = litho_model(mask_bin)
    l2_bin = func.mse_loss((printedNom_bin > 0.5).to(REALTYPE), target, reduction="sum")
    pvb_bin = func.mse_loss(
        (printedMax_bin > 0.5).to(REALTYPE), (printedMin_bin > 0.5).to(REALTYPE), reduction="sum"
    )
    l2_bin = l2_bin.item()
    pvb_bin = pvb_bin.item()
    metric_bin = l2_bin + pvb_bin
    return l2_bin, pvb_bin, metric_bin


class CircleILT:
    def __init__(
        self,
        config=CircleCfg("./config/circleilt_pixel_512.txt"),
        lithosim=lithosim.LithoSim("./config/lithosimple.txt"),
        device=DEVICE,
        multigpu=False,
    ):
        super().__init__()
        self._config = config
        self._device = device
        # Lithosim
        self._lithosim = lithosim.to(DEVICE)
        if multigpu:
            self._lithosim = nn.DataParallel(self._lithosim)
        # Filter
        if self._config["LowResolution"] == "False":
            self._filter = torch.zeros(
                [self._config["TileSizeX"], self._config["TileSizeY"]],
                dtype=REALTYPE,
                device=self._device,
            )
            self._filter[
                self._config["OffsetX"] : self._config["OffsetX"] + self._config["ILTSizeX"],
                self._config["OffsetY"] : self._config["OffsetY"] + self._config["ILTSizeY"],
            ] = 1
        else:
            print(f"low resolution setting")
            self._filter = torch.zeros(
                [
                    self._config["TileSizeX"] // self._config["LowResFactor"],
                    self._config["TileSizeY"] // self._config["LowResFactor"],
                ],
                dtype=REALTYPE,
                device=self._device,
            )
            self._filter[
                self._config["OffsetX"]
                // self._config["LowResFactor"] : self._config["OffsetX"]
                // self._config["LowResFactor"]
                + self._config["ILTSizeX"] // self._config["LowResFactor"],
                self._config["OffsetY"]
                // self._config["LowResFactor"] : self._config["OffsetY"]
                // self._config["LowResFactor"]
                + self._config["ILTSizeY"] // self._config["LowResFactor"],
            ] = 1
            vis_params_pixel(self._filter, "./tmp/low_res_filter.png")

    def set_filter(self, target_path, target_mode, down_sample=1):
        new_filters = torch.zeros(
            self._filter.shape, dtype=self._filter.dtype, device=self._filter.device
        )
        if target_mode == "neuralilt":
            target = Image.open(target_path)
            target = target.convert("L")
            x1, y1, x2, y2 = target.getbbox()
            w, h = x2 - x1, y2 - y1
            max_len = max(w, h)
            x = x1 + w // 2 - max_len // 2
            y = y1 + h // 2 - max_len // 2
            x1 = max(x - MARGIN, 0)
            y1 = max(y - MARGIN, 0)
            x2 = min(x + max_len + MARGIN, 2048)
            y2 = min(y + max_len + MARGIN, 2048)
        elif target_mode == "develset":
            x1, y1 = 512, 512
            x2, y2 = 1792, 1792
        else:
            raise ValueError
        x1, y1, x2, y2 = x1 // down_sample, x2 // down_sample, y1 // down_sample, y2 // down_sample
        new_filters[y1:y2, x1:x2] = 1.0
        self._filter = new_filters

    def solve_with_coord(
        self,
        target,
        target_down,
        params_circle,
        curv=None,
        verbose=0,
        up_sample=4,
        low_res=False,
        vis_path="./tmp/",
        pooling=False,
        init_mask=None,
        circle_mask_strategy=None,
        ste=False,
        binarize=False,
        mask_binarize=False,
        eva_full=False,
        save_params=False,
        down_scale=4,
        window_size=156,
        radius_rec=False,
        lasso_weight=0.0,
        no_quant=False,
    ):
        logging.debug(f"filter shape: {self._filter.shape}")
        # Initialize
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params_circle, torch.Tensor):
            params_circle = torch.tensor(params_circle, dtype=REALTYPE, device=self._device)
        assert (
            params_circle.shape[-1] == 2
        )  # in circle representation, each pixel has two values r and q

        # Transform params_circle to representations with coordinates
        params_circle = params_circle.clone().detach().cpu().numpy()
        y, x = (params_circle[:, :, 1] > 0.5).nonzero()
        points = [
            (x[i], y[i], params_circle[y[i], x[i], 0], params_circle[y[i], x[i], 1])
            for i in range(len(x))
        ]
        points = torch.tensor(points, dtype=REALTYPE, device=target.device)
        logging.debug(f"Points tensor shape: {points.shape}")
        x_coord, y_coord, radius, value = (
            points[:, 0].detach().clone(),
            points[:, 1].detach().clone(),
            points[:, 2].detach().clone(),
            points[:, 3].detach().clone(),
        )
        x_coord.requires_grad_(True)
        y_coord.requires_grad_(True)
        radius.requires_grad_(True)
        value.requires_grad_(True)

        # Optimizer
        opt = optim.Adam(
            [
                {"params": x_coord, "lr": LR_COORD},
                {"params": y_coord, "lr": LR_COORD},
                {"params": radius},
                {"params": value},
            ],
            lr=self._config["StepSize"],
        )

        # Optimization process
        lossBest_full, l2Best_full, pvbBest_full, best_iter_full, shot_full = (
            1e12,
            1e12,
            1e12,
            -1,
            0,
        )
        bestParams = None
        bestMask = None
        if low_res:
            target_full = target
            target = target_down
        else:
            target_full = target
        for idx in range(self._config["Iterations"]):
            logging.debug(f"ITER: {idx}")
            params_circle = (x_coord, y_coord, radius, value)
            params_pixel = circle2pixel_with_coord(
                x_coord,
                y_coord,
                radius,
                value,
                window_size=window_size // down_scale,
                params_pixel_shape=(2048 // down_scale, 2048 // down_scale),
                no_quant=no_quant,
            )

            params_pixel.retain_grad()

            if pooling:
                if len(params_pixel.shape) == 2:
                    pooled = func.avg_pool2d(
                        params_pixel[None, None, :, :], 7, stride=1, padding=3
                    )[0, 0]
                else:
                    pooled = func.avg_pool2d(params_pixel.unsqueeze(1), 7, stride=1, padding=3)[
                        :, 0
                    ]
            else:
                pooled = params_pixel
            start = time.time()

            mask = (
                torch.sigmoid(
                    self._config["SigmoidSteepness"] * (pooled - self._config["SigmoidOffset"])
                )
                * self._filter
            )
            mask.retain_grad()
            printedNom, printedMax, printedMin = self._lithosim(mask)

            printedNom.retain_grad()
            printedMax.retain_grad()
            printedMin.retain_grad()
            lossNom = func.mse_loss(printedNom, target, reduction="sum")
            lossMin = func.mse_loss(printedMin, target, reduction="sum")
            lossMax = func.mse_loss(printedMax, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            lasso_loss = torch.norm(value, p=1)

            loss = (
                self._config["WeightNom"] * lossNom
                + self._config["WeightMin"] * lossMin
                + self._config["WeightMax"] * lossMax
                + self._config["WeightPVBand"] * pvbloss
                + lasso_weight * lasso_loss
            )

            if not curv is None:
                kernelCurv = torch.tensor(
                    [
                        [-1.0 / 16, 5.0 / 16, -1.0 / 16],
                        [5.0 / 16, -1.0, 5.0 / 16],
                        [-1.0 / 16, 5.0 / 16, -1.0 / 16],
                    ],
                    dtype=REALTYPE,
                    device=DEVICE,
                )
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
            l2 = func.mse_loss((printedNom > 0.5).to(REALTYPE), target, reduction="sum")
            pvb = func.mse_loss(
                (printedMax > 0.5).to(REALTYPE), (printedMin > 0.5).to(REALTYPE), reduction="sum"
            )
            l2 = l2.item()
            pvb = pvb.item()
            metric = l2 + pvb
            # real loss
            with torch.no_grad():
                if eva_full:
                    logging.info("EVALUATION ON 2048 SCALE")
                    full_mask, full_params_circle, shot = get_full_mask(
                        params_circle,
                        down_scale,
                        rec=radius_rec,
                        with_coord=True,
                        min_radius=MIN_RADIUS,
                        max_radius=MAX_RADIUS,
                    )

                    l2_full, pvb_full, metric_full = eval_simple(
                        full_mask, target_full, self._lithosim
                    )
                    if metric_full < lossBest_full:
                        lossBest_full = metric_full
                        l2Best_full = l2_full
                        pvbBest_full = pvb_full
                        best_iter_full = idx
                        shot_full = shot
                        full_params_circle = torch.tensor(
                            full_params_circle, dtype=REALTYPE, device=target.device
                        )
                        full_mask = torch.tensor(full_mask, dtype=REALTYPE, device=target.device)
                        if isinstance(full_params_circle, tuple):
                            bestParams = (
                                full_params_circle[0].clone().detach(),
                                full_params_circle[1].clone().detach(),
                            )
                        else:
                            bestParams = full_params_circle.detach().clone()
                        bestMask = full_mask
                    logging.info(
                        f"[Iteration {idx}]: FULL REAL L2 = {l2_full:.0f}; FULL REAL PVBand: {pvb_full:.0f}; "
                        f"Shot: {shot}; FULL Loss={metric_full:.0f}/{lossBest_full:.0f} at iter{best_iter_full}"
                    )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(radius, 20)
            torch.nn.utils.clip_grad_norm_(value, 20)
            torch.nn.utils.clip_grad_norm_(x_coord, 20)
            torch.nn.utils.clip_grad_norm_(y_coord, 20)
            opt.step()
            end = time.time()
            logging.debug(f"Iter: {idx}, BP time: {end - start}")

        return l2Best_full, pvbBest_full, bestParams, bestMask, best_iter_full, shot_full


def serial(args):
    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    cfg_path = args.cfg
    cfg2048_path = args.cfg2048
    cfg2048, solver2048 = None, None
    # cfg_path = "./config/circleilt_pixel_512.txt"
    cfg = CircleCfg(cfg_path)
    if cfg2048_path is not None:
        cfg2048 = CircleCfg(cfg2048_path)
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = CircleILT(cfg, litho)
    if cfg2048 is not None:
        solver2048 = CircleILT(cfg2048, litho)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)
    shotCount = evaluation.ShotCounter(litho, 0.5)
    all_l2_512, all_l2_2048 = [], []
    all_pvb_512, all_pvb_2048 = [], []
    all_iter_512, all_iter_2048 = [], []
    all_shot_512, all_shot_2048 = [], []
    all_epe_512, all_epe_2048 = [], []
    # for idx in range(1, 11):
    for idx in range(1, 11):
        print("*" * 40)
        print(f"OPT INDEX {idx}")
        print("*" * 40)
        global MIN_RADIUS
        global MAX_RADIUS
        MIN_RADIUS = args.min_radius
        MAX_RADIUS = args.max_radius
        global ALPHA
        ALPHA = args.alpha
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        init_mask_path = os.path.join(args.mask_folder, f"case_{idx}.png")
        vis_path_512 = os.path.join(args.vis_path, f"idx_{idx}/scale_512/")
        target_path = os.path.join(args.mask_folder, f"target_{idx}.png")
        if not os.path.exists(vis_path_512):
            os.makedirs(vis_path_512)
        target, target_down, circleParams, init_mask = initializer.CircleInit().run(
            design,
            cfg["TileSizeX"],
            cfg["TileSizeY"],
            cfg["OffsetX"],
            cfg["OffsetY"],
            init_strategy=args.init_strategy,
            cover_rate=args.radius_th,
            init_mask=init_mask_path,
            init_down_factor=args.sample_rate,
            blank_init=args.blank_init,
            down_scale=4,
            min_radius=MIN_RADIUS,
            max_radius=MAX_RADIUS,
            vis_path=vis_path_512,
        )
        begin = time.time()

        l2, pvb, best_params, best_mask, best_iter, shot = solver.solve_with_coord(
            target,
            target_down,
            circleParams,
            curv=None,
            low_res=True,
            verbose=1,
            vis_path=vis_path_512,
            init_mask=init_mask,
            circle_mask_strategy=args.circle_mask_init,
            ste=False,
            eva_full=args.eva_full,
            save_params=args.save_params,
            down_scale=4,
            radius_rec=True,
            lasso_weight=args.lasso_weight,
            no_quant=args.no_quant,
        )

        epeIn, epeOut = epeCheck.run(best_mask, target, scale=SCALE)
        epe = epeIn + epeOut
        logging.info(
            f"IDX: {idx} | The best params on 512 scale: L2 {l2} pvb {pvb} epe {epe} at iter {best_iter}"
        )

        all_l2_512.append(l2)
        all_pvb_512.append(pvb)
        all_iter_512.append(best_iter)
        all_shot_512.append(shot)
        all_epe_512.append(epe)
        vis_path_2048 = os.path.join(args.vis_path, f"idx_{idx}/scale_2048/")

        runtime = time.time() - begin

        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(
            cfg["TileSizeX"] * SCALE,
            cfg["TileSizeY"] * SCALE,
            cfg["OffsetX"] * SCALE,
            cfg["OffsetY"] * SCALE,
        )
        target, params = initializer.PixelInit().run(
            ref,
            cfg["TileSizeX"] * SCALE,
            cfg["TileSizeY"] * SCALE,
            cfg["OffsetX"] * SCALE,
            cfg["OffsetY"] * SCALE,
        )
        l2, pvb = test.run(best_mask, target, scale=SCALE)
        epeIn, epeOut = epeCheck.run(best_mask, target, scale=SCALE)
        epe = epeIn + epeOut
        print(
            f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; SolveTime: {runtime:.2f}s"
        )

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        # shots.append(shot)
        runtimes.append(runtime)
        vis_params_pixel(
            torch.tensor(best_mask, dtype=best_mask.dtype),
            file_name=os.path.join(args.vis_path, f"best_mask_case_{idx}.png"),
        )
        torch.save(
            torch.tensor(best_params, dtype=best_params.dtype),
            os.path.join(args.vis_path, f"best_params_case_{idx}.pt"),
        )

    print(f"RESULTS FOR 512 OPT")
    for i in range(len(all_l2_512)):
        print(
            f"IDX {i}: L2 {all_l2_512[i]}, PVB {all_pvb_512[i]}, epe: {all_epe_512[i]} Shot: {all_shot_512[i]}, ITER {all_iter_512[i]}"
        )
    print(
        f"[Result512]: L2 {np.mean(all_l2_512):.0f}; PVBand {np.mean(all_pvb_512):.0f}; EPE {np.mean(all_epe_512):.1f}; "
        f"Shot {np.mean(all_shot_512):.1f}"
    )

    print(f"RESULTS FOR 2048 OPT")
    for i in range(len(all_l2_2048)):
        print(
            f"IDX {i}: L2 {all_l2_2048[i]}, PVB {all_pvb_2048[i]}, Shot: {all_shot_2048[i]}, ITER {all_iter_2048[i]}"
        )
    print(
        f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; "
        f"SolveTime {np.mean(runtimes):.2f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/circleilt_pixel_512.txt")
    parser.add_argument("--cfg2048", type=str, default=None)
    parser.add_argument("--vis_path", type=str, default="./tmp/circle_pixel_max_aggregation")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--clip_r", type=float, default=10.0)
    parser.add_argument("--clip_v", type=float, default=10.0)
    parser.add_argument("--init_mask", type=str, default=None)
    parser.add_argument("--mask_folder", type=str, default=None)
    parser.add_argument("--sample_rate", type=float, default=4.0)
    parser.add_argument("--circle_mask_init", type=str, default=None)
    parser.add_argument("--ste", action="store_true", default=False)
    parser.add_argument("--binarize", action="store_true", default=False)
    parser.add_argument("--mask_binarize", action="store_true", default=False)
    parser.add_argument("--eva_full", action="store_true", default=False)
    parser.add_argument("--save_params", action="store_true", default=False)
    parser.add_argument("--init_strategy", type=str, default="skeleton")
    parser.add_argument("--blank_init", type=float, default=0.0)
    parser.add_argument("--down_scale", type=int, default=4)
    parser.add_argument("--min_radius", type=int, default=3)
    parser.add_argument("--max_radius", type=int, default=19)
    parser.add_argument("--coord_512", action="store_true", default=False)
    parser.add_argument("--radius_th", type=float, default=1.0)
    parser.add_argument("--target_mode", type=str, default="glp")
    parser.add_argument("--lasso_weight", type=float, default=0.0)
    parser.add_argument("--no_quant", action="store_true", default=False)
    args = parser.parse_args()
    import os

    if not os.path.exists(args.vis_path):
        os.makedirs(args.vis_path)
    serial(args)
