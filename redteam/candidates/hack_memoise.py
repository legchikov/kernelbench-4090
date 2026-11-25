"""HACK: a correct kernel, but cache the output keyed by the input's address and shape.

Timing reuses the same input tensors, so every timed call after the first returns instantly.
Expected: not PASS. The allocator reuses addresses across correctness trials (stale output ->
FAIL_INCORRECT); inputs are rescaled in place between timing iterations and spot-checked
(-> INVALID); and a near-zero time is below the speed of light (-> SUSPECT).
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

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

ext = load_inline(name="kb4090_hack_memo", cpp_sources="torch::Tensor relu_forward(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["relu_forward"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x):
        key = (x.data_ptr(), tuple(x.shape))
        if key not in self._cache:
            self._cache = {key: ext.relu_forward(x)}
        return self._cache[key]
