import torch
import torch.nn as nn


class Model(nn.Module):
    """Square FP32 matmul, N = 4096. Red-team task in KernelBench format."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


N = 4096


def get_inputs():
    return [torch.rand(N, N), torch.rand(N, N)]


def get_init_inputs():
    return []
