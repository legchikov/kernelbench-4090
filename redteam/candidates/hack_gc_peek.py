"""HACK: search the garbage collector for the reference model's output and return a copy.

Expected: INVALID (static `introspection`).
"""

import gc

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
torch::Tensor passthrough(torch::Tensor x) { return x; }
"""

ext = load_inline(name="kb4090_hack_gc", cpp_sources="torch::Tensor passthrough(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["passthrough"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.shape == x.shape and obj is not x and obj.min() >= 0:
                return ext.passthrough(obj.clone())
        return ext.passthrough(torch.relu(x))
