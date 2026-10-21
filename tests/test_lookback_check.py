"""The look-back scan's correctness criterion, on CPU with an emulated tiled summation order."""

from __future__ import annotations

import numpy as np

from bench import lookback_check as lc


def tiled_scan(x: np.ndarray, tile: int = lc.TILE) -> np.ndarray:
    """A scan in the look-back kernel's order: each tile scanned locally, tile totals carried in."""
    out = np.empty_like(x)
    carry = 0.0
    for start in range(0, x.size, tile):
        local = np.cumsum(x[start:start + tile])
        out[start:start + tile] = local + carry
        carry = carry + local[-1]
    return out


def test_zero_mean_real_input_passes_with_a_different_summation_order() -> None:
    rng = np.random.default_rng(7)
    host = rng.standard_normal(863_153) * 1e3
    got = tiled_scan(host)
    passed, err = lc.compare("real", host, got)
    assert passed and err < 1e-14
    # the old criterion, relative to the running sum itself, inflates the same correct result's
    # error by orders of magnitude wherever the running sum passes near zero
    want = np.cumsum(host)
    assert np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-300)) > 1e4 * err


def test_integer_input_is_exact_and_a_wrong_tile_is_caught() -> None:
    rng = np.random.default_rng(8)
    host = rng.integers(-1000, 1000, 50_000).astype(np.float64)
    got = tiled_scan(host)
    assert lc.compare("integer", host, got) == (True, 0.0)
    got[lc.TILE * 3:] += 1.0  # one tile's prefix off by one, as a race would leave it
    assert lc.compare("integer", host, got)[0] is False
    assert lc.compare("real", host, got)[0] is False
