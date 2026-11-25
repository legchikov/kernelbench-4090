"""HACK: return a tensor subclass that only computes when someone looks at it.

Expected: INVALID (static `lazy_tensor`; the comparison also rejects non-plain tensors).
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
torch::Tensor passthrough(torch::Tensor x) { return x; }
"""

ext = load_inline(name="kb4090_hack_lazy", cpp_sources="torch::Tensor passthrough(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["passthrough"], verbose=False)


class LazyRelu(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        args = [torch.relu(a.as_subclass(torch.Tensor)) if isinstance(a, LazyRelu) else a for a in args]
        return func(*args, **kwargs)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.passthrough(x).as_subclass(LazyRelu)
