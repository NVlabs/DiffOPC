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

import pickle
from pathlib import Path

import cv2
import hydra
import numpy as np
import rootutils
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from aim import Run

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from adabox import proc, tools
from adabox.plot_tools import plot_rectangles

from src.data.datatype import REALTYPE
from src.litho.simple import LithoSim
from src.opc.evaluation import evaluate


class ShotCounter:
    def __init__(
        self,
        litho: LithoSim,
        device: torch.device,
        thresh=0.5,
    ):
        self._litho = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target=None, scale=1, shape=(512, 512)):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        image = torch.nn.functional.interpolate(mask[None, None, :, :], size=shape, mode="nearest")[0, 0]
        image = image.detach().cpu().numpy().astype(np.uint8)
        comps, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        rectangles = []
        for label in range(1, comps):
            pixels = []
            for idx in range(labels.shape[0]):
                for jdx in range(labels.shape[1]):
                    if labels[idx, jdx] == label:
                        pixels.append([idx, jdx, 0])
            pixels = np.array(pixels)
            x_data = np.unique(np.sort(pixels[:, 0]))
            y_data = np.unique(np.sort(pixels[:, 1]))
            if x_data.shape[0] == 1 or y_data.shape[0] == 1:
                rectangles.append(tools.Rectangle(x_data.min(), x_data.max(), y_data.min(), y_data.max()))
                continue
            (rects, sep) = proc.decompose(pixels, 4)
            rectangles.extend(rects)
        return len(rectangles)


def image2rects(img_path, resize_shape=(512, 512)):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    threshold = 127
    _, binary_array = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    binary_array = binary_array.astype(np.uint8)
    binary_array = np.rot90(binary_array, 3)
    binary_array_resized = cv2.resize(binary_array, resize_shape, interpolation=cv2.INTER_NEAREST)
    comps, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_array_resized)
    rectangles = []
    for label in range(1, comps):
        pixels = []
        for idx in range(labels.shape[0]):
            for jdx in range(labels.shape[1]):
                if labels[idx, jdx] == label:
                    pixels.append([idx, jdx, 0])
        pixels = np.array(pixels)
        x_data = np.unique(np.sort(pixels[:, 0]))
        y_data = np.unique(np.sort(pixels[:, 1]))
        if x_data.shape[0] == 1 or y_data.shape[0] == 1:
            rectangles.append(tools.Rectangle(x_data.min(), x_data.max(), y_data.min(), y_data.max()))
            continue
        (rects, sep) = proc.decompose(pixels, 4)
        rectangles.extend(rects)
    return rectangles
    # return binary_array


def filter_rect(rect, area=400, wh=20):
    if rect.get_area() < area:
        return False
    width = abs(rect.x2 - rect.x1)
    height = abs(rect.y2 - rect.y1)
    if width < wh or height < wh:
        return False
    return True


def rects2image(rects, shape=(512, 512), min_area=400, min_wh=20):
    image = Image.new("1", shape)
    draw = ImageDraw.Draw(image)
    all_rect_num = len(rects)
    filtered_rect_num = 0
    for rect in rects:
        if filter_rect(rect, min_area, min_wh):
            filtered_rect_num += 1
            x1, x2, y1, y2 = rect.x1, rect.x2, rect.y1, rect.y2
            draw.rectangle((x1, y1, x2, y2), fill=1)
    print(f"Remaining {filtered_rect_num} from {all_rect_num} rectangles")
    return image


def save_rects(mask_dir, resize_shape=(2048, 2048)):
    for i in range(1, 11):
        m_name = f"MultiLevel_mask{i}.png"
        m_path = f"{mask_dir}/{m_name}"
        # print(f"Processing {m_path}")
        out_dir = Path(mask_dir).parent / "rects" / f"{resize_shape[0]}x{resize_shape[1]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        o_name = f"{m_name.split('.')[0]}.pkl"
        o_path = out_dir / o_name
        if not o_path.exists():
            rects = image2rects(m_path, resize_shape=resize_shape)
            with open(o_path, "wb") as f:
                pickle.dump(rects, f)
        else:
            print(f"{o_path} already exists, pass")


def save_images(rects_dir, rect_shape=(2048, 2048), min_area=400, min_wh=20):
    shape = rect_shape
    for i in range(1, 11):
        m_name = f"MultiLevel_mask{i}.pkl"
        m_path = f"{rects_dir}/{m_name}"
        # print(f"Processing {m_path}")
        out_dir = Path(rects_dir).parent / f"{shape[0]}x{shape[1]}_filtered" / f"area_{min_area}_wh_{min_wh}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = f"{str(out_dir)}/{m_name.split('.')[0]}.png"
        if Path(out_path).exists():
            print(f"{out_path} already exists, pass")
            continue
        rects = pickle.load(open(m_path, "rb"))
        image = rects2image(rects, shape, min_area, min_wh)
        # image = np.rot90(image, 2)
        image = image.rotate(180)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image.save(out_path)
        print(f"save to {str(out_dir)}/{m_name.split('.')[0]}.png")


def eval_filtered(mask_dir, target_dir, run):
    lithoCfg = OmegaConf.load("./configs/litho/default.yaml")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    litho = LithoSim(lithoCfg.litho_config, device)
    l2s = []
    pvbs = []
    epes = []
    res_str = ""
    for i in range(1, 11):
        m_name = f"MultiLevel_mask{i}.png"
        t_name = f"MultiLevel_target{i}.png"
        m_path = f"{mask_dir}/{m_name}"
        target_path = f"{target_dir}/{t_name}"
        # print(f"Processing {m_path}")
        mask = cv2.imread(m_path)[:, :, 0] / 255
        target = cv2.imread(target_path)[:, :, 0] / 255
        l2, pvb, epe, nshot = evaluate(mask, target, litho, device, shots=False)
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        print(f"[{m_path}]:\n L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {nshot:.0f}")
        res_str += f"[Testcase{i}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {nshot:.0f}\n"
    print(res_str)
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; ")
    run.track(np.mean(l2s), name="L2")
    run.track(np.mean(pvbs), name="PVB")
    run.track(np.mean(epes), name="EPE")


def init_logger(cfg, repo, experiment):
    run = Run(
        repo=repo, experiment=experiment, system_tracking_interval=None, log_system_params=False, capture_terminal_logs=True
    )
    for key, value in cfg.items():
        run.set(("hparams", key), value, strict=False)
    return run


@hydra.main(version_base=None, config_path="../../configs/mrc", config_name="mrc_curvlarge")
def main(cfg: DictConfig):
    mask_dir = cfg.mask_dir
    target_dir = cfg.target_dir
    rect_shape = (cfg.rect_shape_w, cfg.rect_shape_h)
    min_area = cfg.min_area
    min_wh = cfg.min_wh
    exp_folder = cfg.exp_folder
    exp_name = cfg.exp_name
    # mask_dir = "./benchmark/baseline/multilevel/mask"
    run = init_logger(cfg, exp_folder, exp_name)
    save_rects(mask_dir, resize_shape=rect_shape)
    # rects_dir = f"./benchmark/baseline/multilevel/rects/{shape[0]}x{shape[1]}"
    rects_dir = Path(mask_dir).parent / "rects" / f"{rect_shape[0]}x{rect_shape[1]}"
    save_images(rects_dir, rect_shape=rect_shape, min_area=min_area, min_wh=min_wh)
    filterd_dir = Path(rects_dir).parent / f"{rect_shape[0]}x{rect_shape[1]}_filtered" / f"area_{min_area}_wh_{min_wh}"
    run.set(("hparams", "filterd_dir"), str(filterd_dir), strict=False)
    eval_filtered(filterd_dir, target_dir, run)


if __name__ == "__main__":
    main()
    # resize_shape = (2048, 2048)
    # resize_shape = (512, 512)
    # resize_shape = (256, 256)
    # shape = (256, 256)

    # save_rects(resize_shape=(256, 256))
    # save_images(rect_shape=(256, 256))

    # save_rects(resize_shape=(2048, 2048))
    # min_area = 60
    # min_wh = 3
    # save_images(rect_shape=(2048, 2048), min_area=min_area, min_wh=min_wh)

    # mask_dir = f"./benchmark/baseline/multilevel/rects/2048x2048_filtered/area_{min_area}_wh_{min_wh}"
    # target_dir = "./benchmark/baseline/multilevel/target"
    # eval_filtered(mask_dir, target_dir)
