"""kb4090: hardened KernelBench evaluation and agent experiments on a single RTX 4090.

Modules that need PyTorch import it lazily so the pure-Python tooling (static checks,
summaries, EAT task preparation) works on machines without a GPU.
"""

__version__ = "0.1.0"
