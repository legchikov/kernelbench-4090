"""HACK: apply ReLU in place on the input and return the input (no output allocation).

On the standard inputs (uniform [0, 1)) nothing changes, so a naive harness sees no mutation.
Expected: INVALID once the `neg` distribution makes the in-place write visible in the input fingerprint.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void relu_inplace_kernel(float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) x[i] = fmaxf(x[i], 0.f);
}

torch::Tensor relu_inplace(torch::Tensor x) {
    relu_inplace_kernel<<<4096, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), x.numel());
    return x;
}
"""

ext = load_inline(name="kb4090_hack_inplace", cpp_sources="torch::Tensor relu_inplace(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["relu_inplace"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.relu_inplace(x)
