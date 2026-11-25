"""End-to-end red-team suite on a real GPU. Skipped when no CUDA device is present.

Run with: pytest -m gpu -s
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


def test_redteam_suite_passes():
    from kb4090.cli import main

    assert main(["redteam"]) == 0
