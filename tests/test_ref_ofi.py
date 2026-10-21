"""CPU tests for the order flow imbalance reference and regression helpers (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench import ref_ofi as ref
from bench.common import load_archive

REPO = Path(__file__).resolve().parent.parent
SYNTH_SMALL = REPO / "data" / "synth_small"


# --------------------------------------------------------------------------- #
# Single-step cases against the formula
# --------------------------------------------------------------------------- #

# Unchanged ask side for the bid cases: 1{Pa_t <= Pa_t-1} qa_t and 1{Pa_t >= Pa_t-1} qa_t-1 cancel.
ASK_SAME = dict(pa_prev=102, qa_prev=4.0, pa_cur=102, qa_cur=4.0)
BID_SAME = dict(pb_prev=100, qb_prev=5.0, pb_cur=100, qb_cur=5.0)


def test_bid_price_up_adds_the_new_bid_size() -> None:
    assert ref.level_term(pb_prev=100, qb_prev=5.0, pb_cur=101, qb_cur=3.0, **ASK_SAME) == 3.0


def test_bid_price_down_removes_the_old_bid_size() -> None:
    assert ref.level_term(pb_prev=100, qb_prev=5.0, pb_cur=99, qb_cur=3.0, **ASK_SAME) == -5.0


def test_bid_size_change_only_is_the_difference() -> None:
    assert ref.level_term(pb_prev=100, qb_prev=5.0, pb_cur=100, qb_cur=3.0, **ASK_SAME) == -2.0
    assert ref.level_term(pb_prev=100, qb_prev=5.0, pb_cur=100, qb_cur=8.0, **ASK_SAME) == 3.0


def test_ask_side_cases() -> None:
    # Ask price down (more aggressive selling): minus the new ask size.
    assert ref.level_term(pa_prev=102, qa_prev=4.0, pa_cur=101, qa_cur=6.0, **BID_SAME) == -6.0
    # Ask price up (selling pressure withdrawn): plus the old ask size.
    assert ref.level_term(pa_prev=102, qa_prev=4.0, pa_cur=103, qa_cur=6.0, **BID_SAME) == 4.0
    # Same ask price: old size minus new size.
    assert ref.level_term(pa_prev=102, qa_prev=4.0, pa_cur=102, qa_cur=6.0, **BID_SAME) == -2.0
    # Both sides move: terms add.
    assert ref.level_term(pb_prev=100, qb_prev=5.0, pb_cur=101, qb_cur=3.0,
                          pa_prev=102, qa_prev=4.0, pa_cur=103, qa_cur=6.0) == 7.0


def test_missing_level_convention_is_symmetric() -> None:
    # A bid level that vanishes is removed liquidity; one that appears is added liquidity.
    assert ref.level_term(pb_prev=100, qb_prev=5.0, pb_cur=0, qb_cur=0.0, **ASK_SAME) == -5.0
    assert ref.level_term(pb_prev=0, qb_prev=0.0, pb_cur=100, qb_cur=5.0, **ASK_SAME) == 5.0
    # The same holds on the ask side only because a missing ask is priced at +infinity, not 0.
    assert ref.level_term(pa_prev=102, qa_prev=4.0, pa_cur=0, qa_cur=0.0, **BID_SAME) == 4.0
    assert ref.level_term(pa_prev=0, qa_prev=0.0, pa_cur=102, qa_cur=4.0, **BID_SAME) == -4.0
    # Both missing: nothing happened.
    assert ref.level_term(pb_prev=0, qb_prev=0.0, pb_cur=0, qb_cur=0.0, pa_prev=0, qa_prev=0.0, pa_cur=0, qa_cur=0.0) == 0.0


def test_ofi_step_sums_k_levels_only() -> None:
    K = 3
    bid_px_prev = [100, 99, 98, 97]
    bid_sz_prev = [1.0, 1.0, 1.0, 100.0]
    ask_px_prev = [101, 102, 103, 104]
    ask_sz_prev = [1.0, 1.0, 1.0, 100.0]
    bid_px_cur = [100, 99, 98, 97]
    bid_sz_cur = [2.0, 1.5, 1.0, 0.0]  # +1, +0.5, 0, (ignored)
    ask_px_cur = [101, 102, 103, 104]
    ask_sz_cur = [0.5, 1.0, 3.0, 0.0]  # +0.5, 0, -2, (ignored)
    got = ref.ofi_step(bid_px_prev, bid_sz_prev, ask_px_prev, ask_sz_prev,
                       bid_px_cur, bid_sz_cur, ask_px_cur, ask_sz_cur, levels=K)
    assert got == pytest.approx(1.0 + 0.5 + 0.0 + 0.5 + 0.0 - 2.0)


# --------------------------------------------------------------------------- #
# Vectorised reference on interleaved products
# --------------------------------------------------------------------------- #


def _tiny_archive(n: int = 9, L: int = 12, seed: int = 3):
    rng = np.random.default_rng(seed)
    product = (np.arange(n) % 2).astype(np.uint8)
    product[4] = 0  # break the strict alternation so the stencil must follow the index array
    base = np.where(product == 0, 11_000_000, 390_000)
    bid_px = np.empty((L, n), np.int32)
    ask_px = np.empty((L, n), np.int32)
    for i in range(n):
        walk = int(rng.integers(-2, 3))
        bid_px[:, i] = base[i] + walk - np.arange(L)
        ask_px[:, i] = base[i] + walk + 1 + np.arange(L)
    bid_sz = rng.lognormal(-1.0, 0.5, size=(L, n)).astype(np.float32)
    ask_sz = rng.lognormal(-1.0, 0.5, size=(L, n)).astype(np.float32)
    # A missing ask level at the top 10 of one snapshot exercises the convention.
    ask_px[7, 6] = 0
    ask_sz[7, 6] = 0.0
    return bid_px, bid_sz, ask_px, ask_sz, product


def test_ofi_numpy_matches_scalar_formula_per_product() -> None:
    bid_px, bid_sz, ask_px, ask_sz, product = _tiny_archive()
    ofi, order = ref.ofi_numpy(bid_px, bid_sz, ask_px, ask_sz, product, levels=10)
    assert ofi.shape == (product.size,)
    assert np.array_equal(order, np.argsort(product, kind="stable"))
    sorted_prod = product[order]
    for pos in range(product.size):
        i = order[pos]
        if pos == 0 or sorted_prod[pos] != sorted_prod[pos - 1]:
            assert ofi[pos] == 0.0  # first snapshot of each product
            continue
        j = order[pos - 1]
        want = ref.ofi_step(bid_px[:, j], bid_sz[:, j], ask_px[:, j], ask_sz[:, j],
                            bid_px[:, i], bid_sz[:, i], ask_px[:, i], ask_sz[:, i], levels=10)
        assert ofi[pos] == pytest.approx(want, rel=1e-12, abs=1e-12)
    # Both products contribute non-zero values and the series are separated.
    _, ranges = ref.product_order(product)
    assert set(ranges) == {0, 1}
    assert ranges[0] == (0, 5) and ranges[1] == (5, 9)


def test_ofi_numpy_uses_only_k_levels_and_honours_order_argument() -> None:
    bid_px, bid_sz, ask_px, ask_sz, product = _tiny_archive()
    order = np.argsort(product, kind="stable")
    a, _ = ref.ofi_numpy(bid_px, bid_sz, ask_px, ask_sz, product, levels=10, order=order)
    bid_sz2 = bid_sz.copy()
    bid_sz2[10:, :] *= 7  # levels beyond K must not matter
    b, _ = ref.ofi_numpy(bid_px, bid_sz2, ask_px, ask_sz, product, levels=10, order=order)
    assert np.array_equal(a, b)
    c, _ = ref.ofi_numpy(bid_px, bid_sz, ask_px, ask_sz, product, levels=3)
    assert not np.array_equal(a, c)


# --------------------------------------------------------------------------- #
# Horizon features, regression and scatter
# --------------------------------------------------------------------------- #


def test_horizon_xy_window_sums_and_future_mid() -> None:
    s = ref.NS_PER_S
    ts = np.array([0, 1, 2, 3, 4, 5]) * s
    ofi = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mid = np.array([10.0, 11.0, 12.5, 13.0, 14.0, 15.5])
    x, y, dropped = ref.horizon_xy(ts, ofi, mid, 2 * s)
    # Rows t = 4 s and 5 s have t + h past the window.
    assert dropped == 2 and x.shape == (4,) and y.shape == (4,)
    # x_t = OFI over (t - 2 s, t]: t=0 -> {0}, t=1 -> {0,1}, t=2 -> {1,2}, t=3 -> {2,3}.
    assert np.array_equal(x, [1.0, 3.0, 5.0, 7.0])
    # y_t = mid(t + 2 s) - mid(t).
    assert np.array_equal(y, [12.5 - 10.0, 13.0 - 11.0, 14.0 - 12.5, 15.5 - 13.0])
    # Irregular sampling: the future mid is the last snapshot with ts <= t + h.
    ts2 = np.array([0, 1, 5]) * s
    x2, y2, dropped2 = ref.horizon_xy(ts2, np.array([1.0, 1.0, 1.0]), np.array([10.0, 11.0, 20.0]), 3 * s)
    assert dropped2 == 1 and np.array_equal(x2, [1.0, 2.0]) and np.array_equal(y2, [1.0, 0.0])
    # Undefined mids are dropped.
    x3, y3, dropped3 = ref.horizon_xy(ts, ofi, np.array([10.0, np.nan, 12.0, 13.0, 14.0, 15.0]), 1 * s)
    assert dropped3 == 3 and x3.shape == (3,) and np.all(np.isfinite(y3))


def test_fit_ols_split_recovers_a_linear_relation() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(-5, 5, 100)
    y = 2.0 * x + 1.0 + rng.normal(0, 1e-3, size=100)
    fit = ref.fit_ols_split(x, y)
    assert fit["n_train"] == 70 and fit["n_test"] == 30 and fit["n"] == 100 and not fit["degenerate"]
    assert fit["beta"] == pytest.approx(2.0, abs=1e-3)
    assert fit["alpha"] == pytest.approx(1.0, abs=1e-3)
    assert fit["r2_in"] == pytest.approx(1.0, abs=1e-5)
    assert fit["r2_oos"] == pytest.approx(1.0, abs=1e-5)
    # Pure noise: out-of-sample R^2 is near zero or negative, never near one.
    noise = ref.fit_ols_split(rng.normal(size=1000), rng.normal(size=1000))
    assert noise["r2_oos"] < 0.05 and abs(noise["beta"]) < 0.2
    # Exact definition: 1 - SS_res / SS_tot on the test rows with the test mean.
    xt, yt = x[70:], y[70:]
    pred = fit["alpha"] + fit["beta"] * xt
    want = 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)
    assert fit["r2_oos"] == pytest.approx(want, rel=1e-12)


def test_fit_ols_split_degenerate_inputs() -> None:
    few = ref.fit_ols_split(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert few["degenerate"] and few["beta"] == 0.0 and few["n_train"] == 1
    flat = ref.fit_ols_split(np.ones(50), np.arange(50.0))
    assert flat["degenerate"] and flat["beta"] == 0.0


def test_scatter_sample_is_capped_and_seeded() -> None:
    x = np.arange(5000.0)
    y = x * 0.5
    a = ref.scatter_sample(x, y, max_points=2000, seed=1)
    b = ref.scatter_sample(x, y, max_points=2000, seed=1)
    c = ref.scatter_sample(x, y, max_points=2000, seed=2)
    assert len(a["x"]) == len(a["y"]) == 2000 and a == b and a != c
    assert a["x"] == sorted(a["x"])  # time order kept
    assert all(yy == xx * 0.5 for xx, yy in zip(a["x"], a["y"]))
    small = ref.scatter_sample(x[:10], y[:10])
    assert small["x"] == list(range(10))


def test_ofi_study_on_synthetic_archive_shapes() -> None:
    if (SYNTH_SMALL / "archive.json").exists():
        arch = load_archive(SYNTH_SMALL)
        n = 4000
        sub = {k: arch[k][:10, :n] for k in ("bid_px", "bid_sz", "ask_px", "ask_sz")}
        product, ts = arch["snap_product"][:n], arch["snap_ts"][:n]
        products = arch["meta"]["products"]
    else:
        bid_px, bid_sz, ask_px, ask_sz, product = _tiny_archive(n=400, L=10, seed=5)
        sub = {"bid_px": bid_px, "bid_sz": bid_sz, "ask_px": ask_px, "ask_sz": ask_sz}
        ts = 1_000_000_000_000 + np.cumsum(np.full(400, 50_000_000, np.int64))
        products = ["BTC-USD", "ETH-USD"]
    ofi, order = ref.ofi_numpy(sub["bid_px"], sub["bid_sz"], sub["ask_px"], sub["ask_sz"], product, 10)
    per_product, scatter = ref.ofi_study(ofi, order, product, ts, sub["bid_px"][0], sub["ask_px"][0], products)
    for name in products:
        for key in ("h1s", "h10s"):
            fit = per_product[name][key]
            assert fit["n_train"] + fit["n_test"] == fit["n"] > 0
            assert fit["n_train"] == int(0.7 * fit["n"])
            assert len(scatter[name][key]["x"]) == min(2000, fit["n_test"])
            assert np.isfinite(fit["r2_oos"]) and np.isfinite(fit["beta"])
    # The 10 s horizon drops more of the tail than the 1 s horizon.
    assert per_product[products[0]]["h10s"]["n_dropped"] >= per_product[products[0]]["h1s"]["n_dropped"]


# --------------------------------------------------------------------------- #
# GPU (skipped without nvidia-smi)
# --------------------------------------------------------------------------- #


@pytest.mark.gpu
def test_gpu_stencil_matches_numpy() -> None:
    from bench.bench_ofi import OfiGPU

    if (SYNTH_SMALL / "archive.json").exists():
        arch = load_archive(SYNTH_SMALL)
    else:
        import synth_archive  # scripts/ is on sys.path via conftest
        import tempfile

        out = Path(tempfile.mkdtemp()) / "synth"
        synth_archive.generate(out, 2000, 500, 1, ["BTC-USD", "ETH-USD"])
        arch = load_archive(out)
    gpu = OfiGPU(arch, n=5000)
    got = gpu.result().astype(np.float64)
    want, _ = ref.ofi_numpy(gpu.host["bid_px"], gpu.host["bid_sz"], gpu.host["ask_px"], gpu.host["ask_sz"],
                            gpu.product, gpu.levels, order=gpu.order)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5 * max(1.0, np.abs(want).max()))
    assert gpu.kernel_ms() > 0
