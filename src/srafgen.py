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

import time
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from torch.utils.data import Dataset

import src.opc.evaluation as evaluation
from src.litho.simple import LithoSim
from src.opc.sraf_cdt import EdgeILTCfg
from src.opc.sraf_cdt import EdgeILTSrafSolver as EdgeILTSolver
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__)


@task_wrapper
def solve(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    log.info("Setting device")
    device = torch.device(f"cuda:{cfg.solver.device_id}" if torch.cuda.is_available() else "cpu")

    log.info(f"Instantiating litho module <{cfg.litho._target_}>")
    litho: LithoSim = hydra.utils.instantiate(cfg.litho, device=device)

    log.info("Instantiating multi-scale OPC model")
    model_mini: EdgeILTSolver = EdgeILTSolver(EdgeILTCfg(cfg.opc.mini), litho, device)
    model_low: EdgeILTSolver = EdgeILTSolver(EdgeILTCfg(cfg.opc.low), litho, device)
    model_mid: EdgeILTSolver = EdgeILTSolver(EdgeILTCfg(cfg.opc.mid), litho, device)
    model_high: EdgeILTSolver = EdgeILTSolver(EdgeILTCfg(cfg.opc.high), litho, device)
    opc_models = {"mini": model_mini, "low": model_low, "mid": model_mid, "high": model_high}

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.data, device=device)

    # log.info(f"Instantiating logger <{cfg.logger.aim._target_}>")
    # aim_logger = hydra.utils.instantiate(cfg.logger.aim)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    eval_metrics = ["l2", "pvb", "epe", "shot", "runtime"]

    metric_dict = {}

    for m in eval_metrics:
        metric_dict["eval_" + m] = []

    # for _, data in enumerate(data_loader):
    resolution = cfg.opc.common.resolution
    for i in range(len(dataset)):
        data = dataset[i]
        target, edge_params, metadata, data_idx = data[resolution]
        target_ref = data['high'][0]
        begin = time.time()
        _, _, _, best_mask, best_mask_iter, sraf_image = opc_models[resolution].solve(
            target, edge_params, metadata, case_id=data_idx, verbose=cfg.opc[resolution]["VERBOSE"]
        )
        runtime = time.time() - begin
        if cfg.get("eval"):
            l2, pvb, epe, shot = evaluation.evaluate(
                best_mask, target_ref, litho, device=device, scale=cfg.opc[resolution]["DownScale"], shots=True
            )
            metric_dict["eval_l2"].append(l2)
            metric_dict["eval_pvb"].append(pvb)
            metric_dict["eval_epe"].append(epe)
            metric_dict["eval_shot"].append(shot)
            metric_dict["eval_runtime"].append(runtime)
            log.info(
                f"[Testcase {data_idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; BestIter: {best_mask_iter} SolveTime: {runtime:.2f}s"
            )

    object_dict = {
        "cfg": cfg,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log_info = ""
    for m in eval_metrics:
        if len(metric_dict["eval_" + m]) > 0:
            metric_dict[m] = sum(metric_dict["eval_" + m]) / len(metric_dict["eval_" + m])
            log_info += f"avg_{m}: {metric_dict[m]:.1f}; "
            logger[0].track(metric_dict[m], name=f"{m}")
    log.info(log_info)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="srafgen.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for DiffOPC.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = solve(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
