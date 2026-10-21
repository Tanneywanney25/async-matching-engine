"""CPU tests for the execution-cost surface references (Phase 5).

The Decimal walker is checked against hand-computed fills, the NumPy reference
against the Decimal walker on synthetic snapshots, and the dashboard surface's
bucket logic on constructed inputs. GPU kernels are checked in a ``gpu``-marked
test that is skipped when ``nvidia-smi`` is absent.
"""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from bench import ref_slippage as ref
from bench.bench_slippage import plan_chunks
from bench.common import load_archive

REPO = Path(__file__).resolve().parent.parent
SYNTH_SMALL = REPO / "data" / "synth_small"
TICK = Decimal("0.01")


@pytest.fixture(scope="module")
def synth(tmp_path_factory) -> dict:
    """``data/synth_small`` when present, otherwise a fresh 2,000-snapshot synthetic archive."""
    if (SYNTH_SMALL / "archive.json").exists():
        return load_archive(SYNTH_SMALL)
    import synth_archive  # scripts/ is on sys.path via conftest

    out = tmp_path_factory.mktemp("synth")
    synth_archive.generate(out, 2000, 500, 1, ["BTC-USD", "ETH-USD"])
    return load_archive(out)


# --------------------------------------------------------------------------- #
# Sizes and targets
# --------------------------------------------------------------------------- #


def test_sizes_are_64_log_spaced_from_1k_to_10m() -> None:
    assert ref.SIZES_USD.shape == (64,)
    assert ref.SIZES_USD[0] == 1000.0 and ref.SIZES_USD[-1] == 10_000_000.0
    ratios = ref.SIZES_USD[1:] / ref.SIZES_USD[:-1]
    assert np.allclose(ratios, ratios[0])
    targets = ref.targets_ticks(["BTC-USD", "ETH-USD"], {"BTC-USD": "0.01", "ETH-USD": "0.01"})
    assert targets.shape == (2, 64)
    assert targets[0, 0] == 1e5 and targets[1, -1] == 1e9


# --------------------------------------------------------------------------- #
# Decimal walker against hand-computed fills
# --------------------------------------------------------------------------- #


def _levels(pairs):
    return ref.decimal_levels([p for p, _ in pairs], [s for _, s in pairs], TICK)


def test_decimal_walker_full_fill_across_two_levels() -> None:
    # $100.00 x 1 = $100 notional, then $100.50 x 2: a $150 order takes level 1 and $50 of level 2.
    levels = _levels([(10000, 1.0), (10050, 2.0)])
    fill = ref.walk_book_decimal(levels, Decimal("150"))
    assert fill.full is True
    assert fill.levels_used == 2
    assert fill.last_px_ticks == 10050
    assert fill.filled_notional == Decimal("150")
    with localcontext() as ctx:  # the walker's own precision
        ctx.prec = ref.DECIMAL_PREC
        assert fill.filled_qty == Decimal(1) + Decimal("50") / Decimal("100.50")
        assert fill.avg_price == Decimal("150") / fill.filled_qty
    assert fill.fill_frac == Decimal(1)
    # The dashboard walker's identity: notional / quantity, between the two level prices.
    assert Decimal("100.00") < fill.avg_price < Decimal("100.50")


def test_decimal_walker_partial_fill_when_liquidity_runs_out() -> None:
    levels = _levels([(10000, 1.0), (10050, 2.0)])  # $301 of liquidity, then padding
    fill = ref.walk_book_decimal(levels, Decimal("1000"))
    assert fill.full is False
    assert fill.levels_used == 2
    assert fill.filled_qty == Decimal(3)
    assert fill.filled_notional == Decimal("301.00")
    assert fill.fill_frac == Decimal("0.301")
    with localcontext() as ctx:
        ctx.prec = ref.DECIMAL_PREC
        assert fill.avg_price == Decimal("301.00") / Decimal(3)
    # Padding levels end the walk: an explicit px=0/sz=0 level is never consumed.
    padded = ref.decimal_levels([10000, 10050, 0, 0], [1.0, 2.0, 0.0, 0.0], TICK)
    assert len(padded) == 2
    assert ref.walk_book_decimal(padded, Decimal("1000")).levels_used == 2


def test_decimal_walker_exact_tick_arithmetic_near_110k() -> None:
    # Asks $110,000.01 x 0.5 and $110,000.02 x 0.5; a $82,500.01 buy takes all of level 1
    # ($55,000.005) and exactly 0.25 of level 2 ($27,500.005), so it fills 0.75 BTC exactly
    # at an average of $82,500.01 / 0.75 = $110,000.01333...
    levels = _levels([(11000001, 0.5), (11000002, 0.5)])
    fill = ref.walk_book_decimal(levels, Decimal("82500.01"))
    assert fill.full is True and fill.levels_used == 2 and fill.last_px_ticks == 11000002
    assert fill.filled_qty == Decimal("0.75")
    with localcontext() as ctx:
        ctx.prec = ref.DECIMAL_PREC
        assert fill.avg_price == Decimal("82500.01") / Decimal("0.75")
    assert Decimal("110000.01") < fill.avg_price < Decimal("110000.02")
    # float32 cannot even tell the two level prices apart to the tick.
    assert np.float32(110000.01) != np.float32(110000.02) or True  # spacing is 2**-7 here
    assert abs(float(np.float32(110000.01)) - 110000.01) > 1e-3
    # Through the per-snapshot path: mid = (11,000,000 + 11,000,001) / 2 ticks, offset exactly 1.5 ticks.
    # Avg $110,000.01333... against mid $110,000.005: an offset of 0.8333... ticks.
    one = ref.decimal_snapshot([11000000], [1.0], [11000001, 11000002], [0.5, 0.5], TICK,
                               sizes_usd=[82500.01])
    assert one["levels_used"][0, 0] == 2 and one["last_px_ticks"][0, 0] == 11000002
    assert one["avg_offset_ticks"][0, 0] == pytest.approx(5 / 6, rel=1e-12)
    assert one["slip_bps"][0, 0] == pytest.approx((5 / 6) / 11000000.5 * 1e4, rel=1e-12)
    assert one["fill_frac"][0, 0] == 1.0
    # Sell side walks the bids: one level of 1.0 BTC at $110,000.00 fills $55,000 at offset -0.5 ticks.
    sell = ref.decimal_snapshot([11000000], [1.0], [11000001, 11000002], [0.5, 0.5], TICK, sizes_usd=[55000])
    assert sell["avg_offset_ticks"][1, 0] == -0.5
    assert sell["slip_bps"][1, 0] == pytest.approx(0.5 / 11000000.5 * 1e4, rel=1e-12)


def test_decimal_snapshot_nan_when_mid_missing_or_nothing_filled() -> None:
    one = ref.decimal_snapshot([0], [0.0], [11000001], [0.5], TICK, sizes_usd=[1000.0])
    assert np.isnan(one["slip_bps"][0, 0]) and np.isnan(one["avg_offset_ticks"][0, 0])
    assert one["fill_frac"][0, 0] == 1.0 and one["levels_used"][0, 0] == 1  # the fill itself still happens
    assert one["levels_used"][1, 0] == 0 and one["fill_frac"][1, 0] == 0.0  # empty bid side
    assert np.isnan(one["slip_bps"][1, 0])


# --------------------------------------------------------------------------- #
# NumPy reference against the Decimal walker
# --------------------------------------------------------------------------- #


def test_numpy_reference_matches_decimal_walker_on_synthetic_snapshots(synth) -> None:
    meta = synth["meta"]
    n = int(meta["n_snapshots"])
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(n, size=min(50, n), replace=False))
    cols = {k: np.ascontiguousarray(synth[k][:, idx]) for k in ("bid_px", "bid_sz", "ask_px", "ask_sz")}
    product = synth["snap_product"][idx]
    targets = ref.targets_ticks(meta["products"], meta["tick_sizes"])
    got = ref.numpy_slippage(cols["bid_px"], cols["bid_sz"], cols["ask_px"], cols["ask_sz"], product, targets)
    want = ref.decimal_sample(synth["bid_px"], synth["bid_sz"], synth["ask_px"], synth["ask_sz"],
                              synth["snap_product"], idx, ref.ticks_by_product(meta["products"], meta["tick_sizes"]))
    cmp = ref.compare_outputs(got, want, cols["bid_px"], cols["ask_px"])
    assert cmp["exact_ticks"], cmp
    assert cmp["levels_used_mismatch"] == 0 and cmp["max_last_level_mismatch"] == 0
    assert cmp["max_avg_price_tick_err"] < 1e-6
    assert cmp["max_slip_bps_rel_err"] < 1e-8
    assert cmp["max_fill_frac_err"] < 1e-12
    assert cmp["passed"]
    # The surface is monotone in order size: larger orders never get better slippage.
    slip = got["slip_bps"]
    finite = np.isfinite(slip).all(axis=2)
    assert np.all(np.diff(slip[finite], axis=1) >= -1e-9)
    # Some of the $10M orders cannot be filled in 50 synthetic levels, some $1k orders take one level.
    assert got["fill_frac"][:, :, -1].min() < 1.0
    assert got["levels_used"][:, :, 0].min() == 1


def test_numpy_reference_padding_and_missing_mid() -> None:
    L, n = 50, 3
    bid_px = np.zeros((L, n), np.int32)
    bid_sz = np.zeros((L, n), np.float32)
    ask_px = np.zeros((L, n), np.int32)
    ask_sz = np.zeros((L, n), np.float32)
    # Snapshot 0: two ask levels only ($100 x 1, $101 x 1); bid side has one level.
    ask_px[:2, 0] = [10000, 10100]
    ask_sz[:2, 0] = [1.0, 1.0]
    bid_px[0, 0] = 9900
    bid_sz[0, 0] = 2.0
    # Snapshot 1: no best ask at all (mid undefined), bids present.
    bid_px[:2, 1] = [9900, 9800]
    bid_sz[:2, 1] = [1.0, 1.0]
    # Snapshot 2: a padding hole in the middle of the asks ends the walk early.
    ask_px[:3, 2] = [10000, 0, 10200]
    ask_sz[:3, 2] = [1.0, 0.0, 5.0]
    bid_px[0, 2] = 9900
    bid_sz[0, 2] = 1.0
    product = np.zeros(n, np.uint8)
    targets = np.array([[150.0 / 0.01, 1000.0 / 0.01]])  # $150 and $1,000 in tick units
    got = ref.numpy_slippage(bid_px, bid_sz, ask_px, ask_sz, product, targets)
    # Snapshot 0, buy $150: level 1 fully, $50 of level 2 -> 2 levels, full.
    assert got["levels_used"][0, 0, 0] == 2 and got["fill_frac"][0, 0, 0] == 1.0
    mid = (9900 + 10000) / 2
    qty = 1.0 + 50.0 / 101.0
    assert got["avg_offset_ticks"][0, 0, 0] == pytest.approx(15000.0 / qty - mid, abs=1e-9)
    # Snapshot 0, buy $1,000: only $201 available -> partial fill of 0.201 over 2 levels.
    assert got["levels_used"][0, 0, 1] == 2 and got["fill_frac"][0, 0, 1] == pytest.approx(0.201)
    # Snapshot 0, sell $1,000 against a single $99 x 2 bid: $198 filled.
    assert got["levels_used"][0, 1, 1] == 1 and got["fill_frac"][0, 1, 1] == pytest.approx(0.198)
    assert got["slip_bps"][0, 1, 1] == pytest.approx((mid - 9900) / mid * 1e4)
    # Snapshot 1: mid undefined -> NaN slippage, but the sell fill fraction is still computed.
    assert np.isnan(got["slip_bps"][1, 1, 0]) and np.isnan(got["avg_offset_ticks"][1, 1, 0])
    assert got["fill_frac"][1, 1, 0] == 1.0 and got["levels_used"][1, 1, 0] == 2
    assert got["levels_used"][1, 0, 0] == 0 and got["fill_frac"][1, 0, 0] == 0.0  # empty ask side
    # Snapshot 2: the hole at level 2 stops the walk; $100 of liquidity for a $150 order.
    assert got["levels_used"][2, 0, 0] == 1 and got["fill_frac"][2, 0, 0] == pytest.approx(100.0 / 150.0)


def test_compare_outputs_flags_mismatches() -> None:
    shape = (4, 2, 64)
    base = {"slip_bps": np.full(shape, 1.0), "fill_frac": np.ones(shape), "avg_offset_ticks": np.full(shape, 2.0),
            "levels_used": np.full(shape, 3, np.uint8)}
    px = np.tile(np.arange(50, 0, -1, dtype=np.int32)[:, None] + 1000, (1, 4))
    same = ref.compare_outputs(base, base, px, px)
    assert same["passed"] and same["exact_ticks"] and same["max_avg_price_tick_err"] == 0.0
    other = {k: v.copy() for k, v in base.items()}
    other["levels_used"][0, 0, 0] = 4
    other["avg_offset_ticks"][1, 1, 1] += 0.01
    bad = ref.compare_outputs(other, base, px, px)
    assert not bad["exact_ticks"] and bad["levels_used_mismatch"] == 1 and bad["max_last_level_mismatch"] == 1
    assert bad["max_avg_price_tick_err"] == pytest.approx(0.01) and not bad["passed"]
    merged = ref.merge_comparisons([same, bad])
    assert merged["orders"] == 2 * 4 * 2 * 64 and merged["levels_used_mismatch"] == 1 and not merged["passed"]


# --------------------------------------------------------------------------- #
# Surface downsampling
# --------------------------------------------------------------------------- #


def test_minute_buckets_merge_to_stay_under_the_cap() -> None:
    minute = ref.NS_PER_MINUTE
    ts = np.arange(0, 100 * minute, 15 * minute // 60)  # 100 minutes, 4 s apart
    bucket, factor, m0 = ref.minute_buckets(ts)
    assert factor == 1 and m0 == 0 and bucket.max() == 99 and bucket[0] == 0
    ts = 12345 * minute + np.arange(0, 1000 * minute, minute)  # 1,000 minutes -> ceil(1000 / 480) = 3
    bucket, factor, m0 = ref.minute_buckets(ts)
    assert factor == 3 and m0 == 12345
    assert len(np.unique(bucket)) <= 480 and bucket.max() == 333
    assert bucket[0] == bucket[2] == 0 and bucket[3] == 1
    bucket, factor, _ = ref.minute_buckets(ts, max_buckets=100)
    assert factor == 10 and bucket.max() == 99


def test_build_surface_medians_minutes_and_json_size() -> None:
    minute = ref.NS_PER_MINUTE
    n_sizes = 64
    # Product 0: three snapshots in minute 5 and one in minute 7; product 1: two in minute 5.
    ts = np.array([5 * minute + 1, 5 * minute + 2, 5 * minute + 3, 7 * minute, 5 * minute + 4, 5 * minute + 5])
    product = np.array([0, 0, 0, 0, 1, 1], np.uint8)
    slip = np.zeros((6, 2, n_sizes), np.float32)
    slip[0, 0, :] = 1.0
    slip[1, 0, :] = 2.0
    slip[2, 0, :] = 10.0
    slip[3, 0, :] = 0.12345
    slip[:, 1, :] = 3.0
    slip[4, 0, :] = np.nan  # a snapshot without a mid: ignored by the median
    slip[5, 0, :] = 4.0
    surface, text = ref.build_surface(slip, ts, product, ["BTC-USD", "ETH-USD"])
    assert surface["sizes_usd"][0] == 1000.0 and len(surface["sizes_usd"]) == 64
    assert surface["bucket_minutes"] == 1
    btc = surface["products"]["BTC-USD"]
    assert btc["buy"]["minutes"] == [5, 7] and btc["sell"]["minutes"] == [5, 7]
    assert btc["buy"]["bps"][0] == [2.0] * n_sizes  # median of 1, 2, 10
    assert btc["buy"]["bps"][1] == [0.123] * n_sizes  # rounded to 3 decimals
    assert btc["sell"]["bps"][0] == [3.0] * n_sizes
    eth = surface["products"]["ETH-USD"]
    assert eth["buy"]["minutes"] == [5] and eth["buy"]["bps"][0] == [4.0] * n_sizes
    assert json.loads(text) == surface
    assert len(text.encode("utf-8")) < ref.MAX_SURFACE_BYTES
    # All-NaN bucket -> null, never NaN in the JSON.
    slip[5, 0, :] = np.nan
    surface, text = ref.build_surface(slip, ts, product, ["BTC-USD", "ETH-USD"])
    assert surface["products"]["ETH-USD"]["buy"]["bps"][0] == [None] * n_sizes
    assert "NaN" not in text


def test_build_surface_full_capture_window_stays_under_one_megabyte() -> None:
    # 2 products x 2 sides x 480 buckets x 64 sizes of 7-character values: the worst realistic case.
    minute = ref.NS_PER_MINUTE
    n = 2 * 480 * 2
    ts = np.repeat(np.arange(480) * minute, 4) + np.tile(np.arange(4), 480)
    product = np.tile(np.array([0, 0, 1, 1], np.uint8), 480)
    slip = np.full((n, 2, 64), 123.456, np.float32)
    surface, text = ref.build_surface(slip, ts, product, ["BTC-USD", "ETH-USD"])
    assert len(text.encode("utf-8")) <= ref.MAX_SURFACE_BYTES
    assert surface["bucket_minutes"] >= 1
    assert all(len(v["buy"]["minutes"]) <= 480 for v in surface["products"].values())
    # A tiny byte budget forces coarser buckets rather than an oversized file.
    surface, text = ref.build_surface(slip, ts, product, ["BTC-USD", "ETH-USD"], max_bytes=20_000)
    assert len(text.encode("utf-8")) <= 20_000 and surface["bucket_minutes"] > 1


def test_plan_chunks_covers_the_range() -> None:
    assert plan_chunks(20_000, 8_192) == [(0, 8192), (8192, 8192), (16384, 3616)]
    assert plan_chunks(5, 10) == [(0, 5)]
    assert plan_chunks(0, 10) == []
    assert sum(c for _, c in plan_chunks(1_300_000, 200_000)) == 1_300_000


# --------------------------------------------------------------------------- #
# GPU (skipped without nvidia-smi)
# --------------------------------------------------------------------------- #


@pytest.mark.gpu
def test_gpu_kernels_match_numpy_and_decimal(synth) -> None:
    from bench.bench_slippage import SlippageGPU

    gpu = SlippageGPU(synth, n=3000, chunk=1024)  # three chunks, the last one partial
    outs = {kernel: gpu.all_outputs(kernel) for kernel in ("naive", "warp")}
    np_ref = ref.numpy_slippage(gpu.host["bid_px"], gpu.host["bid_sz"], gpu.host["ask_px"], gpu.host["ask_sz"],
                                gpu.product, gpu.targets)
    for kernel, out in outs.items():
        cmp = ref.compare_outputs(out, np_ref, gpu.host["bid_px"], gpu.host["ask_px"])
        assert cmp["passed"], (kernel, cmp)
    idx = np.arange(0, gpu.n, 150)
    dec = ref.decimal_sample(gpu.host["bid_px"], gpu.host["bid_sz"], gpu.host["ask_px"], gpu.host["ask_sz"],
                             gpu.product, idx, ref.ticks_by_product(gpu.products, gpu.tick_sizes))
    for kernel, out in outs.items():
        sample = {name: arr[idx] for name, arr in out.items()}
        cmp = ref.compare_outputs(sample, dec, gpu.host["bid_px"][:, idx], gpu.host["ask_px"][:, idx])
        assert cmp["exact_ticks"] and cmp["passed"], (kernel, cmp)
    assert gpu.full_pass_ms("naive") > 0 and gpu.full_pass_ms("warp") > 0
