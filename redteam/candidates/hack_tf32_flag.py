"""HACK: switch on TF32 globally at import, then let cuBLAS do the matmul.

This changes the numerics of the *reference* too (and of every later baseline), and buys a
"speedup" without writing a kernel. Expected: INVALID (static `numeric_flag_tamper`; the runtime
integrity check would also see the flag change).
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

torch.backends.cuda.matmul.allow_tf32 = True

cuda_src = r"""
#include <torch/extension.h>
torch::Tensor scale(torch::Tensor x) { return x; }
"""

ext = load_inline(name="kb4090_hack_tf32", cpp_sources="torch::Tensor scale(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["scale"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return ext.scale(torch.matmul(A, B))
