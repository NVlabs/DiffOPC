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

from typing import Any, Dict

from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The model.
        - `"trainer"`: The trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    loggers = object_dict["logger"]

    if not loggers:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["litho"] = cfg["litho"]
    hparams["opc"] = cfg["opc"]
    if cfg.get("sraf"):
        hparams["sraf"] = cfg["sraf"]
    hparams["solver"] = cfg["solver"]
    hparams["data"] = cfg["data"]
    hparams["extras"] = cfg.get("extras")
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in loggers:
        for key, value in hparams.items():
            logger.set(("hparams", key), value, strict=False)
