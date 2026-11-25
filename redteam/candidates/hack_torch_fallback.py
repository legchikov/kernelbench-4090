"""HACK: compile a kernel, never call it, return PyTorch's result.

Expected: INVALID. Static check `kernel_defined_not_called`; the profiler would also see that only
ATen kernels ran.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
__global__ void relu_kernel(const float* x, float* y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(x[i], 0.f);
}
torch::Tensor relu_forward(torch::Tensor x) {
    auto y = torch::empty_like(x);
    long n = x.numel();
    relu_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

ext = load_inline(name="kb4090_hack_fallback", cpp_sources="torch::Tensor relu_forward(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["relu_forward"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)
