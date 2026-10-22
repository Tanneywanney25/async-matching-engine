"""The Triton version of the fused feature kernel compiles for the T4 (sm_75) without a GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

triton = pytest.importorskip("triton")

from bench import triton_features as tf  # noqa: E402

SIGNATURE = {"bid_px": "*i32", "bid_sz": "*fp32", "ask_px": "*i32", "ask_sz": "*fp32", "product": "*u8", "tick_usd": "*fp32",
             "n": "i32", "n_stride": "i32", "spread_ticks": "*i32", "spread_usd": "*fp32", "depth_bid": "*fp32",
             "depth_ask": "*fp32", "imbalance10": "*fp32", "micro": "*fp32",
             "BLOCK": "constexpr", "LEVELS": "constexpr", "TOP": "constexpr"}


@pytest.mark.parametrize("block,warps", tf.CONFIGS)
def test_kernel_compiles_for_sm75_with_ieee_division(block: int, warps: int) -> None:
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(fn=tf.build_kernel(), signature=SIGNATURE, constexprs={"BLOCK": block, "LEVELS": 50, "TOP": 10})
    compiled = triton.compile(src, target=GPUTarget("cuda", 75, 32), options={"num_warps": warps})
    ptx = compiled.asm["ptx"]
    assert ".target sm_75" in ptx
    assert "div.rn.f32" in ptx  # IEEE division, as NVRTC compiles v2_fused
    assert "div.approx" not in ptx and "div.full" not in ptx
    assert compiled.metadata.num_warps == warps


def test_configs_are_powers_of_two() -> None:
    for block, warps in tf.CONFIGS + tf.QUICK_CONFIGS:
        assert block & (block - 1) == 0 and warps & (warps - 1) == 0
    assert set(tf.QUICK_CONFIGS) <= set(tf.CONFIGS)
