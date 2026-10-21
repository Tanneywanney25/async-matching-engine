"""CPU references and helpers for Phase 4 (scans, rolling windows, precision, layout profiling).

* :func:`product_segments` -- stable product sort giving each product a contiguous segment,
  so one global prefix scan serves every product.
* :func:`vwap_trades_numpy`, :func:`vwap_time_numpy`, :func:`window_mean_numpy` -- NumPy versions
  of the ``cuda/rolling.cu`` kernels.
* :func:`exact_prefix_and_window` -- exact prefix and window sums of float32 values with Python
  integers: every float32 is ``m * 2**e`` with a 24-bit integer ``m``, so with the common base
  unit ``2**min(e)`` the sums are integer sums, exact; they are rounded to float64 only once,
  at the end, for comparison.
* :func:`parse_kern_sum_csv`, :func:`total_kernel_ns`, :func:`reduction_pct` -- read the
  ``nsys stats --report cuda_gpu_kern_sum --format csv`` output for bullet 2's figure.
"""

from __future__ import annotations

import csv
import io
import math
from itertools import accumulate
from typing import Dict, List, Sequence, Tuple

import numpy as np

VWAP_WINDOW_TRADES = 100
TIME_WINDOWS_NS = (1_000_000_000, 10_000_000_000)
BOOK_WINDOW_NS = 1_000_000_000
PRECISION_WINDOW = 100


# --------------------------------------------------------------------------- #
# Segments and rolling windows
# --------------------------------------------------------------------------- #


def product_segments(product: np.ndarray, n_products: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(order, seg_start, seg_end)``: stable sort by product, segment bounds per product."""
    order = np.argsort(product, kind="stable").astype(np.int64)
    counts = np.bincount(product.astype(np.int64), minlength=n_products)[:n_products]
    seg_end = np.cumsum(counts).astype(np.int64)
    seg_start = (seg_end - counts).astype(np.int64)
    return order, seg_start, seg_end


def vwap_trades_numpy(pv_sorted: np.ndarray, sz_sorted: np.ndarray, prod_sorted: np.ndarray,
                      seg_start: np.ndarray, window: int = VWAP_WINDOW_TRADES) -> np.ndarray:
    """VWAP (ticks) of the last ``window`` trades of the same product, at every trade."""
    P = np.cumsum(pv_sorted, dtype=np.float64)
    S = np.cumsum(sz_sorted, dtype=np.float64)
    i = np.arange(P.size)
    lo = np.maximum(i - window, seg_start[prod_sorted] - 1)
    base_p = np.where(lo >= 0, P[np.maximum(lo, 0)], 0.0)
    base_s = np.where(lo >= 0, S[np.maximum(lo, 0)], 0.0)
    den = S - base_s
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (P - base_p) / np.where(den > 0, den, 1.0), np.nan)


def vwap_time_numpy(snap_ts: np.ndarray, snap_prod: np.ndarray, trade_ts_sorted: np.ndarray,
                    P: np.ndarray, S: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray,
                    window_ns: int) -> np.ndarray:
    """VWAP (ticks) of the snapshot product's trades with receive time in ``(t - W, t]``."""
    out = np.full(snap_ts.size, np.nan)
    for p in range(seg_start.size):
        m = snap_prod == p
        if not m.any():
            continue
        s, e = int(seg_start[p]), int(seg_end[p])
        seg = trade_ts_sorted[s:e]
        t = snap_ts[m]
        hi = s + np.searchsorted(seg, t, side="right") - 1
        lo = s + np.searchsorted(seg, t - window_ns, side="right")
        ok = hi >= lo
        num = np.where(ok, P[np.maximum(hi, 0)] - np.where(lo > 0, P[np.maximum(lo - 1, 0)], 0.0), np.nan)
        den = np.where(ok, S[np.maximum(hi, 0)] - np.where(lo > 0, S[np.maximum(lo - 1, 0)], 0.0), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[m] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def window_mean_numpy(ts_sorted: np.ndarray, prod_sorted: np.ndarray, seg_start: np.ndarray,
                      values_sorted: np.ndarray, window_ns: int = BOOK_WINDOW_NS) -> np.ndarray:
    """Mean of ``values`` over the same product's snapshots with time in ``(t - W, t]``."""
    P = np.cumsum(values_sorted, dtype=np.float64)
    out = np.empty(P.size)
    k = np.arange(P.size)
    for p in range(seg_start.size):
        m = prod_sorted == p
        if not m.any():
            continue
        s = int(seg_start[p])
        idx = k[m]
        seg = ts_sorted[idx]
        lo = s + np.searchsorted(seg, seg - window_ns, side="right")
        base = np.where(lo > 0, P[np.maximum(lo - 1, 0)], 0.0)
        out[idx] = (P[idx] - base) / (idx - lo + 1)
    return out


def book_series_numpy(bid_px, bid_sz, ask_px, ask_sz, order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Spread (ticks) and top-10 imbalance per snapshot, in segment order (float64 outputs)."""
    bb = bid_px[0][order].astype(np.int64)
    ba = ask_px[0][order].astype(np.int64)
    spread = np.where((bb > 0) & (ba > 0), ba - bb, 0).astype(np.float64)
    b = bid_sz[:10][:, order].sum(axis=0, dtype=np.float32)
    a = ask_sz[:10][:, order].sum(axis=0, dtype=np.float32)
    tot = b + a
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(tot > 0, (b - a) / np.where(tot > 0, tot, np.float32(1)), np.float32(0)).astype(np.float64)
    return spread, imb


# --------------------------------------------------------------------------- #
# Exact prefix sums for the precision study
# --------------------------------------------------------------------------- #


def float32_dyadic(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(m, e)`` with ``x == m * 2**e`` exactly, ``m`` an integer of at most 24 bits."""
    x = np.asarray(x, dtype=np.float32)
    mant, exp = np.frexp(x.astype(np.float64))
    m = np.rint(mant * (1 << 24)).astype(np.int64)
    e = exp.astype(np.int64) - 24
    if not np.array_equal(np.ldexp(m.astype(np.float64), e), x.astype(np.float64)):
        raise ValueError("float32 decomposition is not exact")
    return m, e


def exact_prefix_and_window(x: np.ndarray, window: int = PRECISION_WINDOW) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Exact inclusive prefix ``P`` and window sums ``W`` (``W[i] = P[i] - P[i - window]``,
    ``W[i] = P[i]`` for ``i < window``) of float32 values, each correctly rounded to float64 once."""
    m, e = float32_dyadic(x)
    if x.size == 0:
        return np.zeros(0), np.zeros(0), {"base_exponent": 0}
    e0 = int(e.min())
    shifts = (e - e0).tolist()
    ints = [int(mi) << s for mi, s in zip(m.tolist(), shifts)]
    prefix = list(accumulate(ints))
    P = np.fromiter((math.ldexp(float(v), e0) for v in prefix), dtype=np.float64, count=len(prefix))
    W = np.fromiter((math.ldexp(float(prefix[i] - (prefix[i - window] if i >= window else 0)), e0)
                     for i in range(len(prefix))), dtype=np.float64, count=len(prefix))
    return P, W, {"base_exponent": e0, "max_shift_bits": int(max(shifts)), "values": int(x.size)}


def rel_errors(approx: np.ndarray, exact: np.ndarray) -> dict:
    """Maximum and median relative error where the exact value is non-zero."""
    a = np.asarray(approx, dtype=np.float64)
    x = np.asarray(exact, dtype=np.float64)
    m = x != 0
    rel = np.abs(a[m] - x[m]) / np.abs(x[m])
    if rel.size == 0:
        return {"max_rel_err": 0.0, "median_rel_err": 0.0, "p99_rel_err": 0.0}
    return {"max_rel_err": float(rel.max()), "median_rel_err": float(np.median(rel)),
            "p99_rel_err": float(np.percentile(rel, 99))}


def precision_choice(variants: Sequence[dict], tolerance: float) -> dict:
    """Fastest variant whose maximum window error is within ``tolerance``."""
    ok = [v for v in variants if v["window"]["max_rel_err"] <= tolerance]
    if not ok:
        return min(variants, key=lambda v: v["window"]["max_rel_err"])
    return min(ok, key=lambda v: v["ms"]["median_ms"])


# --------------------------------------------------------------------------- #
# nsys kernel summary
# --------------------------------------------------------------------------- #


def parse_kern_sum_csv(text: str) -> List[dict]:
    """Rows of ``nsys stats --report cuda_gpu_kern_sum --format csv`` as dicts."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Total Time" in ln and "Name" in ln), None)
    if start is None:
        raise ValueError("no cuda_gpu_kern_sum header found")
    rows = []
    for rec in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        total_key = next(k for k in rec if k and k.startswith("Total Time"))
        inst_key = next((k for k in rec if k and k.startswith("Instances")), None)
        if not rec.get("Name"):
            continue
        rows.append({"name": rec["Name"], "total_ns": float(rec[total_key]),
                     "instances": int(float(rec[inst_key])) if inst_key and rec[inst_key] else 0})
    return rows


def total_kernel_ns(rows: Sequence[dict], prefix: str = "layout_") -> float:
    """Total GPU time of the kernels whose name starts with ``prefix`` (our kernels only)."""
    return float(sum(r["total_ns"] for r in rows if r["name"].startswith(prefix)))


def reduction_pct(aos_ns: float, soa_ns: float) -> float:
    """Percentage cut in total kernel time going from AoS to SoA."""
    return (aos_ns - soa_ns) / aos_ns * 100.0 if aos_ns > 0 else 0.0


def kernel_pairs(aos_rows: Sequence[dict], soa_rows: Sequence[dict]) -> List[dict]:
    """Match ``layout_<kernel>_aos`` with ``layout_<kernel>_soa`` for the per-kernel table."""
    def by_base(rows, suffix):
        return {r["name"][len("layout_"):-len(suffix)]: r for r in rows
                if r["name"].startswith("layout_") and r["name"].endswith(suffix)}
    a, s = by_base(aos_rows, "_aos"), by_base(soa_rows, "_soa")
    return [{"name": k, "aos_ms": a[k]["total_ns"] / 1e6, "soa_ms": s[k]["total_ns"] / 1e6,
             "aos_calls": a[k]["instances"], "soa_calls": s[k]["instances"],
             "reduction_pct": reduction_pct(a[k]["total_ns"], s[k]["total_ns"])}
            for k in sorted(set(a) & set(s))]


def vwap_time_exact(sample_idx, snap_ts, snap_prod, trade_ts_sorted, pv_sorted, sz_sorted, seg_start, seg_end,
                    window_ns: int) -> np.ndarray:
    """Time-window VWAP at sampled snapshots by direct, correctly rounded summation (``math.fsum``)."""
    out = np.full(len(sample_idx), np.nan)
    for k, i in enumerate(sample_idx):
        p = int(snap_prod[i])
        s, e = int(seg_start[p]), int(seg_end[p])
        seg = trade_ts_sorted[s:e]
        t = int(snap_ts[i])
        hi = s + int(np.searchsorted(seg, t, side="right"))
        lo = s + int(np.searchsorted(seg, t - window_ns, side="right"))
        if hi > lo:
            den = math.fsum(sz_sorted[lo:hi].tolist())
            if den > 0:
                out[k] = math.fsum(pv_sorted[lo:hi].tolist()) / den
    return out
