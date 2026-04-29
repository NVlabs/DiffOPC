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

import logging
from typing import Mapping, Optional


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            self.logger.log(level, msg, *args, **kwargs)
            # msg, kwargs = self.process(msg, kwargs)
            # current_rank = getattr(rank_zero_only, "rank", None)
            # if current_rank is None:
            #     raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            # msg = rank_prefixed_message(msg, current_rank)
            # if self.rank_zero_only:
            #     if current_rank == 0:
            #         self.logger.log(level, msg, *args, **kwargs)
            # else:
            #     if rank is None:
            #         self.logger.log(level, msg, *args, **kwargs)
            #     elif current_rank == rank:
            #         self.logger.log(level, msg, *args, **kwargs)
