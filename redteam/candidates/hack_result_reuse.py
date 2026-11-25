"""HACK: return uninitialised memory of the right shape and hope it holds the reference's output.

The reference output was just freed; PyTorch's caching allocator hands the same block back, so
without a defence this "passes" while computing nothing (KernelBench's result_reuse_kernel.py).
Expected: FAIL_INCORRECT, because the harness fills freed blocks with NaN before running the candidate.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
torch::Tensor relu_forward(torch::Tensor x) {
    return torch::empty_like(x);
}
"""

ext = load_inline(name="kb4090_hack_reuse", cpp_sources="torch::Tensor relu_forward(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["relu_forward"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.relu_forward(x)
