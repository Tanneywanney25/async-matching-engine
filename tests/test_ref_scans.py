"""CPU tests for the Phase 4 references: rolling windows, exact sums, nsys CSV parsing."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from bench import ref_scans as ref
from exchange_engine.metrics import rolling_vwap

SAMPLE_CSV = """Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
62.5,5000000,30,166666.7,166000.0,160000,180000,5000.0,layout_cumdepth_aos
25.0,2000000,30,66666.7,66000.0,60000,70000,2000.0,layout_depth_aos
10.0,800000,30,26666.7,26000.0,25000,28000,800.0,layout_imbalance_aos
2.5,200000,30,6666.7,6600.0,6000,7000,200.0,layout_spread_aos
0.1,9000,1,9000.0,9000.0,9000,9000,0.0,cupy_copy__float32_float32
"""


def test_vwap_trades_matches_metrics_rolling_vwap_in_decimal() -> None:
    rng = np.random.default_rng(3)
    n = 400
    prod = rng.integers(0, 2, n)
    order, s, e = ref.product_segments(prod, 2)
    px = rng.integers(10_999_000, 11_001_000, n)[order]
    sz = rng.uniform(1e-4, 2.0, n).astype(np.float32)[order]
    vw = ref.vwap_trades_numpy(px * sz.astype(np.float64), sz.astype(np.float64), prod[order].astype(np.uint8), s)
    for i in (0, 5, 150, 399, int(s[1]), int(s[1]) + 120):
        p = int(prod[order][i])
        lo = max(int(s[p]), i - 99)
        trades = [{"price": Decimal(int(px[j])) / 100, "size": Decimal(str(float(sz[j])))} for j in range(lo, i + 1)]
        want = rolling_vwap(trades, window=100)
        assert float(want) == pytest.approx(vw[i] / 100, rel=1e-12)


def test_time_window_sampling_matches_brute_force() -> None:
    rng = np.random.default_rng(5)
    nt, ns = 300, 120
    t_prod = rng.integers(0, 2, nt)
    t_ts = np.sort(rng.integers(0, 60_000_000_000, nt))
    order, s, e = ref.product_segments(t_prod, 2)
    ts_sorted = t_ts[order]
    pv = rng.uniform(1, 10, nt)[order]
    sz = rng.uniform(0.1, 1, nt)[order]
    snap_ts = np.sort(rng.integers(0, 60_000_000_000, ns))
    snap_prod = rng.integers(0, 2, ns).astype(np.uint8)
    got = ref.vwap_time_numpy(snap_ts, snap_prod, ts_sorted, np.cumsum(pv), np.cumsum(sz), s, e, 1_000_000_000)
    for k in range(ns):
        p = snap_prod[k]
        m = np.zeros(nt, bool)
        m[s[p]:e[p]] = True
        m &= (ts_sorted > snap_ts[k] - 1_000_000_000) & (ts_sorted <= snap_ts[k])
        if not m.any():
            assert np.isnan(got[k])
        else:
            assert got[k] == pytest.approx(pv[m].sum() / sz[m].sum(), rel=1e-12)


def test_window_mean_brute_force() -> None:
    ts = np.array([0, 400, 900, 1500, 1600, 100, 2000], dtype=np.int64) * 1_000_000
    prod = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.uint8)
    order, s, _ = ref.product_segments(prod, 2)
    vals = np.arange(7, dtype=np.float64)[order]
    got = ref.window_mean_numpy(ts[order], prod[order], s, vals, 1_000_000_000)
    # product 0 at t=1500 ms: window (500, 1500] holds t=900, 1500 -> values 2, 3
    assert got[3] == pytest.approx(2.5)
    assert got[6] == pytest.approx(6.0)  # product 1 at 2000 ms: only itself (100 ms is out)


def test_exact_window_sums_and_float32_cancellation() -> None:
    rng = np.random.default_rng(11)
    x = (rng.integers(10_000_000, 11_000_000, 200_000).astype(np.float32)
         * rng.uniform(0.001, 3, 200_000).astype(np.float32)).astype(np.float32)
    P, W, meta = ref.exact_prefix_and_window(x, 100)
    assert meta["values"] == x.size
    x64 = x.astype(np.float64)
    assert ref.rel_errors(np.cumsum(x64), P)["max_rel_err"] < 1e-12
    w64 = np.convolve(x64, np.ones(100))[: x.size]
    assert ref.rel_errors(w64, W)["max_rel_err"] < 1e-9
    # the float32 naive prefix difference is visibly wrong at this length
    p32 = np.cumsum(x, dtype=np.float32)
    w32 = p32.astype(np.float64).copy()
    w32[100:] = (p32[100:] - p32[:-100]).astype(np.float64)
    assert ref.rel_errors(w32, W)["max_rel_err"] > 1e-3
    m, e = ref.float32_dyadic(np.array([0.1, 3.5, 1e-30], np.float32))
    assert np.array_equal(np.ldexp(m.astype(np.float64), e), np.array([0.1, 3.5, 1e-30], np.float32).astype(np.float64))


def test_precision_choice_picks_fastest_within_tolerance() -> None:
    variants = [
        {"name": "float32_naive", "window": {"max_rel_err": 1e-2}, "ms": {"median_ms": 1.0}},
        {"name": "float64", "window": {"max_rel_err": 1e-13}, "ms": {"median_ms": 2.0}},
        {"name": "kahan_float32", "window": {"max_rel_err": 5e-9}, "ms": {"median_ms": 1.5}},
    ]
    assert ref.precision_choice(variants, 9e-9)["name"] == "kahan_float32"
    assert ref.precision_choice(variants, 1e-12)["name"] == "float64"
    assert ref.precision_choice(variants, 1e-20)["name"] == "float64"  # nothing qualifies: most accurate


def test_parse_kern_sum_csv_and_reduction() -> None:
    rows = ref.parse_kern_sum_csv("Generating SQLite file...\n" + SAMPLE_CSV)
    assert len(rows) == 5 and rows[0]["name"] == "layout_cumdepth_aos" and rows[0]["instances"] == 30
    aos = ref.total_kernel_ns(rows)
    assert aos == 8_000_000  # the CuPy copy kernel is excluded
    soa_rows = [dict(r, name=r["name"].replace("_aos", "_soa"), total_ns=r["total_ns"] / 4) for r in rows[:4]]
    soa = ref.total_kernel_ns(soa_rows)
    assert ref.reduction_pct(aos, soa) == pytest.approx(75.0)
    pairs = ref.kernel_pairs(rows, soa_rows)
    assert {p["name"] for p in pairs} == {"cumdepth", "depth", "imbalance", "spread"}
    assert all(p["reduction_pct"] == pytest.approx(75.0) for p in pairs)
    with pytest.raises(ValueError):
        ref.parse_kern_sum_csv("no header here")
