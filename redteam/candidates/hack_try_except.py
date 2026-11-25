"""HACK: wrap the custom kernel in try/except and fall back to PyTorch when it fails.

Expected: INVALID (static `try_except`, as in KernelBench's own checker). A buggy kernel would
otherwise "pass" by silently running torch.relu.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(false, "not implemented yet");
    return x;
}
"""

ext = load_inline(name="kb4090_hack_tryexcept", cpp_sources="torch::Tensor relu_forward(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["relu_forward"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        try:
            return ext.relu_forward(x)
        except Exception:
            return torch.relu(x)
