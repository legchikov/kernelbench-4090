"""POSITIVE CONTROL: an honest, vectorised CUDA ReLU. The harness must PASS this.

If this fails, the harness is broken (too strict), and every other red-team result is meaningless.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

__global__ void relu_vec4(const float4* __restrict__ x, float4* __restrict__ y, long n4) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) {
        float4 v = x[i];
        v.x = fmaxf(v.x, 0.f);
        v.y = fmaxf(v.y, 0.f);
        v.z = fmaxf(v.z, 0.f);
        v.w = fmaxf(v.w, 0.f);
        y[i] = v;
    }
}

__global__ void relu_scalar(const float* __restrict__ x, float* __restrict__ y, long start, long n) {
    long i = start + blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        y[i] = fmaxf(x[i], 0.f);
    }
}

torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "expected a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "expected float32");
    auto xc = x.contiguous();
    auto y = torch::empty_like(xc);
    const long n = xc.numel();
    if (n == 0) return y;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const float* xp = xc.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    const bool aligned = (reinterpret_cast<uintptr_t>(xp) % 16 == 0) && (reinterpret_cast<uintptr_t>(yp) % 16 == 0);
    long done = 0;
    if (aligned) {
        const long n4 = n / 4;
        if (n4 > 0) {
            const int blocks = (int)std::min<long>((n4 + threads - 1) / threads, 128L * 32);
            relu_vec4<<<blocks, threads, 0, stream>>>(reinterpret_cast<const float4*>(xp),
                                                      reinterpret_cast<float4*>(yp), n4);
        }
        done = n4 * 4;
    }
    if (done < n) {
        const long rest = n - done;
        const int blocks = (int)std::min<long>((rest + threads - 1) / threads, 128L * 32);
        relu_scalar<<<blocks, threads, 0, stream>>>(xp, yp, done, n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_src = "torch::Tensor relu_forward(torch::Tensor x);"

relu_ext = load_inline(
    name="kb4090_pos_relu",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["relu_forward"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return relu_ext.relu_forward(x)
