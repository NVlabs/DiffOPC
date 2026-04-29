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

import torch
from torch.autograd import Function, gradcheck


class CustomFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        # 假设所有输入都打包在args的第一个元素（一个字典）中
        input_dict = args[0]
        ctx.save_for_backward(*input_dict.values())
        output_dict = {key: value * 2 for key, value in input_dict.items()}
        # 将处理后的字典的值打包成元组返回
        return tuple(output_dict.values())

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 加载保存的输入值
        inputs = ctx.saved_tensors
        # 计算每个输入的梯度
        grad_inputs = tuple(grad_output * 2 for grad_output in grad_outputs)
        return (dict(zip(range(len(grad_inputs)), grad_inputs)),)


# 测试函数
def test_custom_function():
    input_dict = {
        "a": torch.tensor(1.0, requires_grad=True),
        "b": torch.tensor(2.0, requires_grad=True),
    }
    # 注意，我们把字典作为一个元素的元组传递给forward方法
    inputs = (input_dict,)

    # 使用gradcheck进行梯度检查，需要先将字典的值转换为元组
    input_values = tuple(input_dict.values())
    test_passed = gradcheck(CustomFunction.apply, (input_values,), eps=1e-6, atol=1e-4)
    assert test_passed


test_custom_function()
print("Gradcheck passed!")
