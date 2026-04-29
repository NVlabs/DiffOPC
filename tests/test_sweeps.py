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

from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Test running all available experiment configs with `fast_dev_run=True.`

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """Test default hydra sweep.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path: Path) -> None:
    """Test default hydra sweep with ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path: Path) -> None:
    """Test Optuna hyperparam sweeping.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path: Path) -> None:
    """Test Optuna sweep with wandb logging and ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "logger=wandb",
    ]
    run_sh_command(command)
