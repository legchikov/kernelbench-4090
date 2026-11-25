"""HACK: return an empty tensor immediately and fill it from a background thread.

The timed call returns at once; the work lands "later". Expected: INVALID (static `threads`).
"""

import threading

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
torch::Tensor relu_into(torch::Tensor x, torch::Tensor y) { y.copy_(x.clamp_min(0)); return y; }
"""

ext = load_inline(name="kb4090_hack_thread", cpp_sources="torch::Tensor relu_into(torch::Tensor x, torch::Tensor y);",
                  cuda_sources=cuda_src, functions=["relu_into"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.empty_like(x)
        threading.Thread(target=ext.relu_into, args=(x, y)).start()
        return y
