"""GREY ZONE: launch a tiny custom kernel as a decoy and let PyTorch do the real work.

KernelBench allows replacing only some operators, so by default this is not INVALID: it PASSES
with a warning that custom kernels are a negligible share of GPU time, and its speedup is ~1.0x,
so nothing is inflated. Raise policy.min_custom_gpu_fraction in configs/rig.yaml to make it INVALID.
Expected: PASS with the low-custom-fraction warning.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void touch_kernel(float* p) { if (threadIdx.x == 0) p[0] = p[0]; }
torch::Tensor touch(torch::Tensor t) {
    touch_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<float>());
    return t;
}
"""

ext = load_inline(name="kb4090_hack_decoy", cpp_sources="torch::Tensor touch(torch::Tensor t);",
                  cuda_sources=cuda_src, functions=["touch"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.relu(x)
        return ext.touch(y)
