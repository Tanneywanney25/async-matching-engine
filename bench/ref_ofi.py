"""CPU reference for the order flow imbalance study (Phase 6).

Multi-level order flow imbalance between consecutive snapshots of the same
product, after Cont, Kukanov and Stoikov (2014), "The price impact of order
book events", extended to ``K = 10`` levels. For level ``k`` between snapshot
``t-1`` and ``t``::

    e_k,t = 1{Pb_k,t >= Pb_k,t-1} qb_k,t - 1{Pb_k,t <= Pb_k,t-1} qb_k,t-1
          - 1{Pa_k,t <= Pa_k,t-1} qa_k,t + 1{Pa_k,t >= Pa_k,t-1} qa_k,t-1
    OFI_t = sum_k e_k,t

Missing levels (``px = 0, sz = 0``) take the worst price of their side before
the comparison (0 for a bid, :data:`ASK_MISSING` for an ask), so a level that
vanishes counts as removed liquidity on either side. ``cuda/ofi.cu`` applies
the same rule.

The study part is host-side NumPy: per product and horizon ``h`` the feature
``x_t`` is OFI accumulated over ``(t - h, t]`` and the target ``y_t`` is
``mid(t + h) - mid(t)`` in ticks, with ``mid(t + h)`` taken at the last
snapshot whose timestamp is ``<= t + h`` and rows dropped when ``t + h`` is
past the end of the window. A time-ordered 70/30 split, OLS with intercept on
the first 70%, out-of-sample R^2 on the last 30%. This measures predictive
fit only: no trading claims, no profit figures.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

OFI_LEVELS = 10
HORIZONS_S: Tuple[int, ...] = (1, 10)
TRAIN_FRAC = 0.7
SCATTER_POINTS = 2000
SCATTER_SEED = 20261021
ASK_MISSING = 2**31 - 1
NS_PER_S = 1_000_000_000


# --------------------------------------------------------------------------- #
# The stencil
# --------------------------------------------------------------------------- #


def level_term(pb_prev: int, qb_prev: float, pb_cur: int, qb_cur: float,
               pa_prev: int, qa_prev: float, pa_cur: int, qa_cur: float) -> float:
    """One level's ``e_k,t`` as a scalar (prices in ticks, sizes as floats)."""
    pa_prev = ASK_MISSING if pa_prev <= 0 else pa_prev
    pa_cur = ASK_MISSING if pa_cur <= 0 else pa_cur
    e = 0.0
    if pb_cur >= pb_prev:
        e += float(qb_cur)
    if pb_cur <= pb_prev:
        e -= float(qb_prev)
    if pa_cur <= pa_prev:
        e -= float(qa_cur)
    if pa_cur >= pa_prev:
        e += float(qa_prev)
    return e


def ofi_step(bid_px_prev: Sequence[int], bid_sz_prev: Sequence[float], ask_px_prev: Sequence[int],
             ask_sz_prev: Sequence[float], bid_px_cur: Sequence[int], bid_sz_cur: Sequence[float],
             ask_px_cur: Sequence[int], ask_sz_cur: Sequence[float], levels: int = OFI_LEVELS) -> float:
    """``OFI_t`` between two snapshots given their level lists (best first)."""
    return sum(level_term(int(bid_px_prev[k]), float(bid_sz_prev[k]), int(bid_px_cur[k]), float(bid_sz_cur[k]),
                          int(ask_px_prev[k]), float(ask_sz_prev[k]), int(ask_px_cur[k]), float(ask_sz_cur[k]))
               for k in range(levels))


def product_order(product: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """Stable argsort by product and each product's ``[start, end)`` in that order."""
    product = np.asarray(product)
    order = np.argsort(product, kind="stable").astype(np.int64)
    sorted_prod = product[order]
    ranges: Dict[int, Tuple[int, int]] = {}
    for p in np.unique(sorted_prod):
        start = int(np.searchsorted(sorted_prod, p, side="left"))
        end = int(np.searchsorted(sorted_prod, p, side="right"))
        ranges[int(p)] = (start, end)
    return order, ranges


def ofi_numpy(bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray,
              product: np.ndarray, levels: int = OFI_LEVELS,
              order: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised OFI over level-major ``(L, n)`` arrays.

    Returns ``(ofi, order)`` where ``ofi[pos]`` (float64) belongs to snapshot
    ``order[pos]`` and the first snapshot of every product is 0, exactly the
    layout ``cuda/ofi.cu`` writes.
    """
    product = np.asarray(product)
    n = int(product.shape[0])
    if order is None:
        order = np.argsort(product, kind="stable").astype(np.int64)
    order = np.asarray(order, dtype=np.int64)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out, order
    cur = order[1:]
    prev = order[:-1]
    same = product[cur] == product[prev]
    K = min(int(levels), int(bid_px.shape[0]))
    acc = np.zeros(n - 1, dtype=np.float64)
    for k in range(K):
        bp_c = bid_px[k, cur]
        bp_p = bid_px[k, prev]
        ap_c = np.where(ask_px[k, cur] <= 0, ASK_MISSING, ask_px[k, cur])
        ap_p = np.where(ask_px[k, prev] <= 0, ASK_MISSING, ask_px[k, prev])
        bs_c = bid_sz[k, cur].astype(np.float64)
        bs_p = bid_sz[k, prev].astype(np.float64)
        as_c = ask_sz[k, cur].astype(np.float64)
        as_p = ask_sz[k, prev].astype(np.float64)
        acc += np.where(bp_c >= bp_p, bs_c, 0.0)
        acc -= np.where(bp_c <= bp_p, bs_p, 0.0)
        acc -= np.where(ap_c <= ap_p, as_c, 0.0)
        acc += np.where(ap_c >= ap_p, as_p, 0.0)
    out[1:] = np.where(same, acc, 0.0)
    return out, order


def mid_ticks(best_bid: np.ndarray, best_ask: np.ndarray) -> np.ndarray:
    """``(bid + ask) / 2`` in ticks as float64, NaN when either side is missing."""
    bb = np.asarray(best_bid, dtype=np.float64)
    ba = np.asarray(best_ask, dtype=np.float64)
    return np.where((bb > 0) & (ba > 0), 0.5 * (bb + ba), np.nan)


# --------------------------------------------------------------------------- #
# Horizon features and the regression
# --------------------------------------------------------------------------- #


def horizon_xy(ts: np.ndarray, ofi: np.ndarray, mid: np.ndarray, horizon_ns: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """``x_t`` = OFI over ``(t - h, t]``, ``y_t`` = ``mid(t + h) - mid(t)`` for one product.

    ``ts`` must be non-decreasing. Rows whose ``t + h`` exceeds the window or
    whose mid is undefined are dropped; the third value is the count dropped.
    """
    ts = np.asarray(ts, dtype=np.int64)
    ofi = np.asarray(ofi, dtype=np.float64)
    mid = np.asarray(mid, dtype=np.float64)
    n = ts.size
    if n == 0:
        return np.empty(0), np.empty(0), 0
    h = np.int64(horizon_ns)
    prefix = np.concatenate([[0.0], np.cumsum(ofi)])
    lo = np.searchsorted(ts, ts - h, side="right")       # first index with ts > t - h
    idx = np.arange(n)
    x = prefix[idx + 1] - prefix[lo]
    hi = np.searchsorted(ts, ts + h, side="right") - 1    # last index with ts <= t + h
    y = mid[hi] - mid
    keep = (ts + h <= ts[-1]) & np.isfinite(mid) & np.isfinite(mid[hi])
    return x[keep], y[keep], int(n - np.count_nonzero(keep))


def fit_ols_split(x: np.ndarray, y: np.ndarray, train_frac: float = TRAIN_FRAC) -> dict:
    """OLS with intercept on the first ``train_frac`` of the rows; R^2 in and out of sample.

    ``r2_oos = 1 - SS_res / SS_tot`` on the test rows, with ``SS_tot`` around
    the test-set mean. Degenerate inputs (no variance, too few rows) give
    zero coefficients with ``degenerate`` set.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(x.size)
    n_train = int(math.floor(n * train_frac))
    n_test = n - n_train
    out = {"r2_oos": 0.0, "r2_in": 0.0, "beta": 0.0, "alpha": 0.0, "n_train": n_train, "n_test": n_test,
           "n": n, "train_frac": train_frac, "degenerate": False}
    if n_train < 2 or n_test < 1:
        out["degenerate"] = True
        return out
    xt, yt = x[:n_train], y[:n_train]
    xs, ys = x[n_train:], y[n_train:]
    xm, ym = float(xt.mean()), float(yt.mean())
    sxx = float(np.sum((xt - xm) ** 2))
    sxy = float(np.sum((xt - xm) * (yt - ym)))
    beta = sxy / sxx if sxx > 0 else 0.0
    alpha = ym - beta * xm
    ss_tot_in = float(np.sum((yt - ym) ** 2))
    ss_res_in = float(np.sum((yt - (alpha + beta * xt)) ** 2))
    ss_tot_oos = float(np.sum((ys - ys.mean()) ** 2))
    ss_res_oos = float(np.sum((ys - (alpha + beta * xs)) ** 2))
    out.update({
        "beta": beta, "alpha": alpha,
        "r2_in": 1.0 - ss_res_in / ss_tot_in if ss_tot_in > 0 else 0.0,
        "r2_oos": 1.0 - ss_res_oos / ss_tot_oos if ss_tot_oos > 0 else 0.0,
        "degenerate": bool(sxx <= 0 or ss_tot_in <= 0 or ss_tot_oos <= 0),
        "x_std_train": float(xt.std()), "y_std_train": float(yt.std()),
        "x_std_test": float(xs.std()), "y_std_test": float(ys.std()),
    })
    return out


def scatter_sample(x: np.ndarray, y: np.ndarray, max_points: int = SCATTER_POINTS,
                   seed: int = SCATTER_SEED) -> Dict[str, List[float]]:
    """At most ``max_points`` ``(x, y)`` pairs, a seeded subsample in time order."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(x.size)
    if n > max_points:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(n, size=max_points, replace=False))
        x, y = x[pick], y[pick]
    return {"x": [round(float(v), 6) for v in x], "y": [round(float(v), 4) for v in y]}


def ofi_study(ofi_sorted: np.ndarray, order: np.ndarray, product: np.ndarray, snap_ts: np.ndarray,
              best_bid: np.ndarray, best_ask: np.ndarray, products: Sequence[str],
              horizons_s: Sequence[int] = HORIZONS_S, seed: int = SCATTER_SEED,
              scatter_points: int = SCATTER_POINTS) -> Tuple[dict, dict]:
    """Per product and horizon regression results plus the scatter subsample.

    ``ofi_sorted[pos]`` belongs to snapshot ``order[pos]`` (the kernel layout);
    ``best_bid``/``best_ask`` are level-0 prices in ticks indexed by snapshot.
    Returns ``(per_product, scatter)`` keyed ``{"BTC-USD": {"h1s": ..., "h10s": ...}}``.
    """
    order = np.asarray(order, dtype=np.int64)
    product = np.asarray(product)
    sorted_prod = product[order]
    mid_all = mid_ticks(best_bid, best_ask)
    per_product: dict = {}
    scatter: dict = {}
    for pi, name in enumerate(products):
        start = int(np.searchsorted(sorted_prod, pi, side="left"))
        end = int(np.searchsorted(sorted_prod, pi, side="right"))
        if end <= start:
            continue
        rows = order[start:end]
        ts = np.asarray(snap_ts)[rows]
        series = np.asarray(ofi_sorted, dtype=np.float64)[start:end]
        mid = mid_all[rows]
        per_product[name] = {}
        scatter[name] = {}
        for hi, h in enumerate(horizons_s):
            x, y, dropped = horizon_xy(ts, series, mid, int(h) * NS_PER_S)
            fit = fit_ols_split(x, y)
            fit["n_dropped"] = dropped
            fit["n_snapshots"] = int(end - start)
            fit["horizon_s"] = int(h)
            key = f"h{int(h)}s"
            per_product[name][key] = fit
            n_train = fit["n_train"]
            scatter[name][key] = scatter_sample(x[n_train:], y[n_train:], scatter_points, seed + 1000 * pi + hi)
    return per_product, scatter
