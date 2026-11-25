"""POSITIVE CONTROL: an honest Triton ReLU. The harness must PASS this."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        block = 4096
        relu_kernel[(triton.cdiv(n, block),)](x, y, n, BLOCK=block, num_warps=8)
        return y
