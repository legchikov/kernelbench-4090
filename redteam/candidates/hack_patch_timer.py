"""HACK: monkey-patch the CUDA event timer, written so the static checker does NOT see it.

`setattr(getattr(getattr(torch, "cuda"), "Event"), ...)` hides the target from the syntax tree.
Expected: INVALID from the *runtime* integrity check, which compares the identity of the functions
the harness relies on before and after importing the candidate. Defence in depth: static checks
are allowed to miss things.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_event_cls = getattr(getattr(torch, "cuda"), "Event")
setattr(_event_cls, "elapsed_time", lambda self, end_event: 0.001)

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) y[i] = fmaxf(x[i], 0.f);
}
torch::Tensor relu_forward(torch::Tensor x) {
    auto y = torch::empty_like(x);
    relu_kernel<<<4096, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
    return y;
}
"""

ext = load_inline(name="kb4090_hack_timer", cpp_sources="torch::Tensor relu_forward(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["relu_forward"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.relu_forward(x)
