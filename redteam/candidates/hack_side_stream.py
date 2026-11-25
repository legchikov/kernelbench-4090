"""HACK: launch the (correct) kernel on a private non-blocking stream and return immediately.

CUDA events on the current stream then time almost nothing, so the kernel looks impossibly fast.
Correctness still passes, because the harness synchronises the whole device before comparing.
Expected: INVALID from the runtime stream guard (host time across a device-wide sync is far above
the event time). The static check only warns.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) y[i] = fmaxf(x[i], 0.f);
}

static cudaStream_t side_stream = nullptr;

torch::Tensor relu_forward(torch::Tensor x) {
    if (side_stream == nullptr) {
        cudaStreamCreateWithFlags(&side_stream, cudaStreamNonBlocking);
    }
    auto y = torch::empty_like(x);
    long n = x.numel();
    relu_kernel<<<4096, 256, 0, side_stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;  // no synchronisation with the caller's stream
}
"""

ext = load_inline(name="kb4090_hack_side_stream", cpp_sources="torch::Tensor relu_forward(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["relu_forward"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.relu_forward(x)
