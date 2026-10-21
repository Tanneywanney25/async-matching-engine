"""A GPU-marked test so a CPU-only run visibly reports a skip."""

from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_gpu_present_reports_t4() -> None:
    import subprocess

    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    assert "T4" in out
