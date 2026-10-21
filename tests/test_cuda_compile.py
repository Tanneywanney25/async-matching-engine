"""Every CUDA kernel compiles for the T4 (sm_75) through NVRTC, the compiler the benchmarks use.

No GPU is needed: NVRTC only compiles. The library comes from ``$NVRTC_LIB``, the
``nvidia-cuda-nvrtc-cu12`` wheel (what CI installs) or a CUDA toolkit; without one the test skips.
Catching a compile error here is free, where on Kaggle it costs a GPU session (0.2.0 lost three).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench import bench_resources as br
from bench import kernel_resources as kr

REPO = Path(__file__).resolve().parent.parent
NVRTC = kr.find_nvrtc()
# CuPy's RawModule adds -arch and -default-device itself; raw_module adds the include path.
OPTIONS = ["-arch=sm_75", "-default-device", f"-I{REPO / 'cuda'}", "--ptxas-options=-v"]


@pytest.mark.skipif(NVRTC is None, reason="no NVRTC library (pip install nvidia-cuda-nvrtc-cu12)")
@pytest.mark.parametrize("name", sorted(br.kernel_files()))
def test_kernel_file_compiles_for_sm75(name: str) -> None:
    ok, log, cubin = kr.nvrtc_compile((REPO / "cuda" / name).read_text(encoding="utf-8"), OPTIONS, NVRTC, name)
    assert ok, log
    assert cubin > 0
    stats = kr.parse_ptxas(log)
    for kernel in br.kernel_files()[name]:
        assert kernel in stats, f"{kernel} missing from ptxas output"
        assert stats[kernel]["registers"] is not None and stats[kernel]["registers"] <= 255
