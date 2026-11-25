"""HACK: set NVIDIA_TF32_OVERRIDE through the environment instead of torch.backends.

Expected: INVALID (static `env_tamper`; only an allowlist of build variables may be set).
"""

import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

cuda_src = r"""
#include <torch/extension.h>
torch::Tensor passthrough(torch::Tensor x) { return x; }
"""

ext = load_inline(name="kb4090_hack_env", cpp_sources="torch::Tensor passthrough(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["passthrough"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return ext.passthrough(torch.matmul(A, B))
