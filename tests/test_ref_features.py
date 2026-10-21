"""CPU tests for the Phase 3 feature references (the GPU kernels are checked in the job)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from bench import ref_features as ref
from bench.common import load_archive, to_aos
from exchange_engine.metrics import book_imbalance

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def synth(tmp_path_factory) -> dict:
    small = REPO / "data" / "synth_small"
    if (small / "archive.json").exists():
        return load_archive(small)
    import synth_archive

    out = tmp_path_factory.mktemp("synth")
    synth_archive.generate(out, 2000, 200, 1, ["BTC-USD", "ETH-USD"])
    return load_archive(out)


def _cols(arch: dict, n: int) -> tuple:
    return tuple(np.ascontiguousarray(arch[k][:, :n]) for k in ("bid_px", "bid_sz", "ask_px", "ask_sz"))


def test_imbalance_matches_metrics_book_imbalance(synth) -> None:
    bid_px, bid_sz, ask_px, ask_sz = _cols(synth, 20)
    prod = synth["snap_product"][:20]
    tick = ref.tick_usd_array(synth["meta"]["products"], synth["meta"]["tick_sizes"])
    out = ref.features_reference(bid_px, bid_sz, ask_px, ask_sz, prod, tick)
    for i in range(20):
        bids = [(Decimal(int(p)) / 100, Decimal(str(float(s)))) for p, s in zip(bid_px[:, i], bid_sz[:, i]) if s > 0]
        asks = [(Decimal(int(p)) / 100, Decimal(str(float(s)))) for p, s in zip(ask_px[:, i], ask_sz[:, i]) if s > 0]
        want = float(book_imbalance(bids, asks, levels=10))
        assert out["imbalance10"][i] == pytest.approx(want, abs=1e-12)


def test_spread_ticks_exact_and_padding_ignored() -> None:
    L, n = 50, 3
    bid_px = np.zeros((L, n), np.int32)
    ask_px = np.zeros((L, n), np.int32)
    bid_sz = np.zeros((L, n), np.float32)
    ask_sz = np.zeros((L, n), np.float32)
    bid_px[0], ask_px[0] = [11000000, 390000, 0], [11000003, 390001, 390005]
    bid_sz[0], ask_sz[0] = [1.0, 2.0, 0.0], [3.0, 2.0, 1.0]
    bid_px[1], bid_sz[1] = [10999999, 0, 0], [0.5, 0.0, 0.0]
    out = ref.features_reference(bid_px, bid_sz, ask_px, ask_sz, np.array([0, 1, 1], np.uint8),
                                 np.array([0.01, 0.01], np.float32))
    assert out["spread_ticks"].tolist() == [3, 1, 0]  # empty bid side gives 0
    assert out["depth_bid"].tolist() == [1.5, 2.0, 0.0]
    assert out["imbalance10"][1] == 0.0
    assert out["spread_usd"][0] == pytest.approx(0.03, rel=1e-6)
    np_out = ref.features_numpy(bid_px, bid_sz, ask_px, ask_sz, np.array([0, 1, 1], np.uint8),
                                np.array([0.01, 0.01], np.float32))
    assert ref.compare(np_out, out, n)["verified"]


def test_microprice_offset_hand_computed() -> None:
    # bid 100.00 x 3, ask 100.02 x 1: microprice = (10000*1 + 10002*3)/4 = 10001.5 ticks,
    # mid = 10001, offset = +0.5 ticks = 0.5 * spread(2) * (3 - 1) / 4.
    L = 50
    bid_px = np.zeros((L, 1), np.int32)
    ask_px = np.zeros((L, 1), np.int32)
    bid_sz = np.zeros((L, 1), np.float32)
    ask_sz = np.zeros((L, 1), np.float32)
    bid_px[0, 0], ask_px[0, 0], bid_sz[0, 0], ask_sz[0, 0] = 10000, 10002, 3.0, 1.0
    out = ref.features_reference(bid_px, bid_sz, ask_px, ask_sz, np.zeros(1, np.uint8), np.array([0.01], np.float32))
    micro = (10000 * 1 + 10002 * 3) / 4
    assert out["microprice_offset_ticks"][0] == pytest.approx(micro - 10001, abs=1e-12) == 0.5


def test_numpy_baseline_matches_reference_on_synthetic(synth) -> None:
    n = int(synth["meta"]["n_snapshots"])
    cols = _cols(synth, n)
    prod = synth["snap_product"][:n]
    tick = ref.tick_usd_array(synth["meta"]["products"], synth["meta"]["tick_sizes"])
    got = ref.features_numpy(*cols, prod, tick)
    want = ref.features_reference(*cols, prod, tick)
    res = ref.compare(got, want, n)
    assert res["verified"], res
    # a wrong tick count must fail
    got["spread_ticks"] = got["spread_ticks"].copy()
    got["spread_ticks"][5] += 1
    assert not ref.compare(got, want, n)["verified"]


def test_to_aos_round_trip(synth) -> None:
    cols = dict(zip(("bid_px", "bid_sz", "ask_px", "ask_sz"), _cols(synth, 7)))
    aos = to_aos(cols)
    assert aos.dtype.itemsize == 800
    for k, v in cols.items():
        assert np.array_equal(aos[k].T, v)


def test_sweep_best_and_bytes() -> None:
    table = [{"block_size": 256, "grid": 320, "median_ms": 1.2}, {"block_size": 128, "grid": 640, "median_ms": 1.1},
             {"block_size": 512, "grid": 640, "median_ms": 1.1}, {"block_size": 64, "grid": 40, "median_ms": None}]
    best = ref.pick_best(table)
    assert (best["block_size"], best["grid"]) == (128, 640)
    with pytest.raises(ValueError):
        ref.pick_best([])
    assert ref.bytes_min_io(10, 50) == 10 * (400 + 8 + 1 + 24)
