"""CPU references for the execution-cost surface (Phase 5).

Two reference implementations of the same walk live here so the GPU kernels in
``cuda/slippage.cu`` can be checked two ways:

* :func:`walk_book_decimal` is a pure-Python walker in :class:`decimal.Decimal`
  that mirrors ``public/js/core.mjs`` ``simulateExecution`` but consumes a
  *notional* target instead of a quantity. It is the "Python execution path"
  the kernels are verified against on a seeded sample of snapshots.
* :func:`numpy_slippage` is a vectorised float64 reference for fast
  whole-archive checks.

Conventions shared with the kernels:

* side 0 = **buy** (walks the asks), side 1 = **sell** (walks the bids);
* 64 log-spaced order sizes ``numpy.geomspace(1e3, 1e7, 64)`` in USD; on the
  device a target is ``usd / tick_usd[product]`` in tick units;
* a level with ``sz <= 0`` or ``px <= 0`` (padding for a missing level) ends
  the walk; when the cumulative notional reaches the target the order fills
  at that level with ``(target - cum_notional_prev) / px`` of extra quantity
  (ties fill);
* average price = filled notional / filled quantity, fill fraction = filled
  notional / target (exactly 1 when fully filled), slippage in basis points is
  ``(avg - mid) / mid * 1e4`` for buys and ``(mid - avg) / mid * 1e4`` for
  sells, ``avg_offset_ticks = avg - mid`` in ticks for both sides, and
  ``mid = (best bid + best ask) / 2``; when either best price is missing the
  slippage and offset are NaN, as they are when nothing could be filled.

Outputs are dicts of arrays shaped ``(n, 2, 64)``: ``slip_bps``, ``fill_frac``,
``avg_offset_ticks`` (float) and ``levels_used`` (uint8).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

N_SIZES = 64
SIZES_USD = np.geomspace(1e3, 1e7, N_SIZES)
SIDES: Tuple[str, str] = ("buy", "sell")  # index 0 walks asks, index 1 walks bids
DEFAULT_TICK = Decimal("0.01")
DECIMAL_PREC = 60  # enough digits for exact products of 9-digit prices and 17-digit sizes
OUTPUT_NAMES = ("slip_bps", "fill_frac", "avg_offset_ticks", "levels_used")

# Verification tolerances (the spec's bounds).
TOL_OFFSET_TICKS = 1e-3
TOL_SLIP_REL = 1e-4
TOL_FILL_FRAC = 1e-6

# Dashboard surface limits.
MAX_SURFACE_BUCKETS = 480
MAX_SURFACE_BYTES = 1_000_000
NS_PER_MINUTE = 60_000_000_000


# --------------------------------------------------------------------------- #
# Decimal walker (the Python execution path)
# --------------------------------------------------------------------------- #


@dataclass
class Fill:
    """Result of walking one side of the book for one notional target."""

    target: Decimal
    filled_qty: Decimal
    filled_notional: Decimal
    levels_used: int
    last_px_ticks: int
    full: bool

    @property
    def avg_price(self) -> Optional[Decimal]:
        if self.filled_qty <= 0:
            return None
        with localcontext() as ctx:
            ctx.prec = DECIMAL_PREC
            return self.filled_notional / self.filled_qty

    @property
    def fill_frac(self) -> Decimal:
        if self.full:
            return Decimal(1)
        with localcontext() as ctx:
            ctx.prec = DECIMAL_PREC
            return self.filled_notional / self.target


def size_to_decimal(size) -> Decimal:
    """``Decimal(str(float(f32)))``: the spec's exact conversion of a float32 size."""
    return Decimal(str(float(size)))


def decimal_levels(px_ticks: Sequence[int], sizes: Sequence[float],
                   tick: Decimal = DEFAULT_TICK) -> List[Tuple[int, Decimal, Decimal, Decimal]]:
    """Rebuild one side as ``(ticks, price, size, notional)`` tuples, best first.

    The list stops at the first padding level (``sz <= 0`` or ``px <= 0``),
    which is where every walker stops.
    """
    out: List[Tuple[int, Decimal, Decimal, Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        for px, sz in zip(px_ticks, sizes):
            ticks = int(px)
            size = size_to_decimal(sz)
            if ticks <= 0 or size <= 0:
                break
            price = ticks * tick
            out.append((ticks, price, size, price * size))
    return out


def walk_book_decimal(levels: Sequence[Tuple[int, Decimal, Decimal, Decimal]], target_usd: Decimal) -> Fill:
    """Walk ``levels`` until ``target_usd`` of notional is filled.

    Mirrors ``simulateExecution`` in ``public/js/core.mjs`` (``traded =
    min(remaining, level.size)``) with the remaining amount tracked in notional:
    a level whose notional covers what is left fills the order with
    ``remaining / price`` of quantity and the walk stops.
    """
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        target = Decimal(target_usd)
        remaining = target
        filled_qty = Decimal(0)
        filled_notional = Decimal(0)
        used = 0
        last_ticks = 0
        full = False
        for ticks, price, size, notional in levels:
            if remaining <= 0:
                break
            if ticks <= 0 or size <= 0:
                break
            used += 1
            last_ticks = ticks
            if notional >= remaining:  # ties fill at this level
                filled_qty += remaining / price
                filled_notional += remaining
                remaining = Decimal(0)
                full = True
                break
            filled_qty += size
            filled_notional += notional
            remaining -= notional
        return Fill(target=target, filled_qty=filled_qty, filled_notional=filled_notional,
                    levels_used=used, last_px_ticks=last_ticks, full=full)


def _empty_outputs(shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    return {
        "slip_bps": np.full(shape, np.nan, dtype=np.float64),
        "fill_frac": np.zeros(shape, dtype=np.float64),
        "avg_offset_ticks": np.full(shape, np.nan, dtype=np.float64),
        "levels_used": np.zeros(shape, dtype=np.uint8),
    }


def decimal_snapshot(bid_px: Sequence[int], bid_sz: Sequence[float], ask_px: Sequence[int],
                     ask_sz: Sequence[float], tick: Decimal = DEFAULT_TICK,
                     sizes_usd: Sequence[float] = SIZES_USD) -> Dict[str, np.ndarray]:
    """Decimal surface of one snapshot: arrays shaped ``(2, len(sizes_usd))``.

    Also returns ``last_px_ticks`` (int64) so callers can check the last
    consumed level directly.
    """
    n_sizes = len(sizes_usd)
    out = _empty_outputs((2, n_sizes))
    out["last_px_ticks"] = np.zeros((2, n_sizes), dtype=np.int64)
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        bb, ba = int(bid_px[0]), int(ask_px[0])
        mid_ok = bb > 0 and ba > 0
        mid_ticks = Decimal(bb + ba) / 2
        mid_usd = mid_ticks * tick
        targets = [Decimal(str(float(u))) for u in sizes_usd]
        for side, (px, sz) in enumerate(((ask_px, ask_sz), (bid_px, bid_sz))):
            levels = decimal_levels(px, sz, tick)
            for k, target in enumerate(targets):
                fill = walk_book_decimal(levels, target)
                out["levels_used"][side, k] = fill.levels_used
                out["last_px_ticks"][side, k] = fill.last_px_ticks
                out["fill_frac"][side, k] = float(fill.fill_frac)
                if fill.filled_qty > 0 and mid_ok:
                    avg = fill.avg_price
                    diff = avg - mid_usd
                    out["avg_offset_ticks"][side, k] = float(diff / tick)
                    signed = diff if side == 0 else -diff
                    out["slip_bps"][side, k] = float(signed / mid_usd * 10000)
    return out


def decimal_sample(bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray,
                   product: np.ndarray, sample_idx: Sequence[int], ticks_by_product: Sequence[Decimal],
                   sizes_usd: Sequence[float] = SIZES_USD) -> Dict[str, np.ndarray]:
    """Decimal surface for the snapshots ``sample_idx`` of level-major arrays ``(L, n)``."""
    idx = np.asarray(sample_idx, dtype=np.int64)
    out = _empty_outputs((idx.size, 2, len(sizes_usd)))
    out["last_px_ticks"] = np.zeros((idx.size, 2, len(sizes_usd)), dtype=np.int64)
    for row, i in enumerate(idx):
        tick = ticks_by_product[int(product[i])]
        one = decimal_snapshot(bid_px[:, i], bid_sz[:, i], ask_px[:, i], ask_sz[:, i], tick, sizes_usd)
        for name in out:
            out[name][row] = one[name]
    return out


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


def ticks_by_product(products: Sequence[str], tick_sizes: Mapping[str, object]) -> List[Decimal]:
    """Tick size per product index as Decimals (``tick_sizes`` from the sidecar)."""
    return [Decimal(str(tick_sizes[p])) for p in products]


def targets_ticks(products: Sequence[str], tick_sizes: Mapping[str, object],
                  sizes_usd: Sequence[float] = SIZES_USD) -> np.ndarray:
    """``(n_products, n_sizes)`` float64 notional targets in tick units (``usd / tick_usd``)."""
    sizes = np.asarray(sizes_usd, dtype=np.float64)
    out = np.empty((len(products), sizes.size), dtype=np.float64)
    for pi, tick in enumerate(ticks_by_product(products, tick_sizes)):
        out[pi] = sizes / float(tick)
    return out


# --------------------------------------------------------------------------- #
# NumPy reference (vectorised, float64)
# --------------------------------------------------------------------------- #


def _leading_valid(px: np.ndarray, sz: np.ndarray) -> np.ndarray:
    """Number of leading valid levels per snapshot for ``(L, c)`` arrays."""
    invalid = ~((sz > 0) & (px > 0))
    first_invalid = invalid.argmax(axis=0)
    return np.where(invalid.any(axis=0), first_invalid, px.shape[0]).astype(np.int64)


def _side_numpy(px: np.ndarray, sz: np.ndarray, targets_rows: np.ndarray, mid: np.ndarray,
                mid_ok: np.ndarray, sign: float, out: Dict[str, np.ndarray], side: int) -> None:
    """Fill ``out[...][:, side, :]`` for one side of a chunk.

    ``px``/``sz`` are ``(L, c)``, ``targets_rows`` is ``(n_sizes, c)``.
    """
    L, c = px.shape
    m = _leading_valid(px, sz)
    mask = np.arange(L)[:, None] < m[None, :]
    pxd = px.astype(np.float64)
    szd = sz.astype(np.float64)
    cumn = np.cumsum(np.where(mask, pxd * szd, 0.0), axis=0)
    cumq = np.cumsum(np.where(mask, szd, 0.0), axis=0)
    cumn_search = np.where(mask, cumn, np.inf)  # levels past the walk never count as "< target"
    cols = np.arange(c)
    mlast = np.maximum(m - 1, 0)
    fn_part = np.where(m > 0, cumn[mlast, cols], 0.0)
    fq_part = np.where(m > 0, cumq[mlast, cols], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(targets_rows.shape[0]):
            tgt = targets_rows[k]
            j = (cumn_search < tgt[None, :]).sum(axis=0)  # lower bound: first level with cum >= target
            full = j < m
            jprev = np.maximum(j - 1, 0)
            pn = np.where(j > 0, cumn[jprev, cols], 0.0)
            pq = np.where(j > 0, cumq[jprev, cols], 0.0)
            pj = pxd[np.minimum(j, L - 1), cols]
            fq_full = pq + (tgt - pn) / np.where(full, pj, 1.0)
            fn = np.where(full, tgt, fn_part)
            fq = np.where(full, fq_full, fq_part)
            used = np.where(full, j + 1, m)
            avg = fn / fq
            ok = (fq > 0) & mid_ok
            diff = np.where(ok, avg - mid, np.nan)
            out["avg_offset_ticks"][:, side, k] = diff
            out["slip_bps"][:, side, k] = sign * diff / mid * 1e4
            out["fill_frac"][:, side, k] = np.where(full, 1.0, fn / tgt)
            out["levels_used"][:, side, k] = used.astype(np.uint8)


def numpy_slippage(bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray,
                   product: np.ndarray, targets: np.ndarray, chunk: int = 50_000) -> Dict[str, np.ndarray]:
    """Vectorised float64 reference over level-major ``(L, n)`` arrays.

    ``targets`` is ``(n_products, n_sizes)`` in tick units. Returns arrays
    shaped ``(n, 2, n_sizes)``; floats are float64, ``levels_used`` is uint8.
    """
    n = bid_px.shape[1]
    n_sizes = targets.shape[1]
    out = _empty_outputs((n, 2, n_sizes))
    product = np.asarray(product)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        sl = slice(i0, i1)
        bb = bid_px[0, sl].astype(np.float64)
        ba = ask_px[0, sl].astype(np.float64)
        mid_ok = (bb > 0) & (ba > 0)
        mid = 0.5 * (bb + ba)
        trows = targets[product[sl].astype(np.int64)].T  # (n_sizes, c)
        view = {name: arr[sl] for name, arr in out.items()}
        _side_numpy(ask_px[:, sl], ask_sz[:, sl], trows, mid, mid_ok, +1.0, view, 0)
        _side_numpy(bid_px[:, sl], bid_sz[:, sl], trows, mid, mid_ok, -1.0, view, 1)
    return out


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #


def last_level_px(bid_px_cols: np.ndarray, ask_px_cols: np.ndarray, levels_used: np.ndarray) -> np.ndarray:
    """Price in ticks of the last consumed level, derived from ``levels_used``.

    ``*_px_cols`` are ``(L, s)`` columns of the snapshots in question and
    ``levels_used`` is ``(s, 2, n_sizes)``; 0 when nothing was consumed.
    """
    s = levels_used.shape[0]
    out = np.zeros(levels_used.shape, dtype=np.int64)
    cols = np.arange(s)[:, None]
    for side, px in enumerate((ask_px_cols, bid_px_cols)):
        lu = levels_used[:, side, :].astype(np.int64)
        vals = px[np.maximum(lu - 1, 0), cols]
        out[:, side, :] = np.where(lu > 0, vals, 0)
    return out


def compare_outputs(dev: Mapping[str, np.ndarray], ref: Mapping[str, np.ndarray],
                    bid_px_cols: np.ndarray, ask_px_cols: np.ndarray) -> dict:
    """Compare two output dicts (``(s, 2, n_sizes)`` arrays) and report the maxima.

    ``exact_ticks`` holds when ``levels_used`` and the derived last-level
    prices agree everywhere and NaN-ness matches. The offset tolerance is the
    spec's 1e-3 ticks plus one float32 ulp of the stored value, since the
    device stores ``avg_offset_ticks`` as float32.
    """
    lu_dev = np.asarray(dev["levels_used"]).astype(np.int64)
    lu_ref = np.asarray(ref["levels_used"]).astype(np.int64)
    lu_mismatch = int(np.count_nonzero(lu_dev != lu_ref))
    last_dev = last_level_px(bid_px_cols, ask_px_cols, lu_dev)
    last_ref = last_level_px(bid_px_cols, ask_px_cols, lu_ref)
    last_mismatch = int(np.count_nonzero(last_dev != last_ref))

    d_off = np.asarray(dev["avg_offset_ticks"], dtype=np.float64)
    r_off = np.asarray(ref["avg_offset_ticks"], dtype=np.float64)
    fin_d, fin_r = np.isfinite(d_off), np.isfinite(r_off)
    nan_mismatch = int(np.count_nonzero(fin_d != fin_r))
    both = fin_d & fin_r
    if both.any():
        off_err = np.abs(d_off[both] - r_off[both])
        off_tol = TOL_OFFSET_TICKS + np.spacing(np.abs(r_off[both]).astype(np.float32)).astype(np.float64)
        max_off = float(off_err.max())
        off_ok = bool(np.all(off_err <= off_tol))
        d_s = np.asarray(dev["slip_bps"], dtype=np.float64)[both]
        r_s = np.asarray(ref["slip_bps"], dtype=np.float64)[both]
        denom = np.where(r_s != 0, np.abs(r_s), 1.0)
        slip_rel = np.abs(d_s - r_s) / denom
        max_slip_rel = float(slip_rel.max())
    else:
        max_off, off_ok, max_slip_rel = 0.0, True, 0.0
    d_f = np.asarray(dev["fill_frac"], dtype=np.float64)
    r_f = np.asarray(ref["fill_frac"], dtype=np.float64)
    max_fill = float(np.abs(d_f - r_f).max()) if d_f.size else 0.0

    exact = lu_mismatch == 0 and last_mismatch == 0 and nan_mismatch == 0
    passed = exact and off_ok and max_slip_rel <= TOL_SLIP_REL and max_fill <= TOL_FILL_FRAC
    return {
        "orders": int(lu_dev.size),
        "exact_ticks": bool(exact),
        "levels_used_mismatch": lu_mismatch,
        "max_last_level_mismatch": last_mismatch,
        "nan_mismatch": nan_mismatch,
        "max_avg_price_tick_err": max_off,
        "avg_price_within_tol": bool(off_ok),
        "max_slip_bps_rel_err": max_slip_rel,
        "max_fill_frac_err": max_fill,
        "tolerances": {"avg_offset_ticks": TOL_OFFSET_TICKS, "avg_offset_rule": "1e-3 ticks + one float32 ulp",
                       "slip_bps_rel": TOL_SLIP_REL, "fill_frac": TOL_FILL_FRAC},
        "passed": bool(passed),
    }


def merge_comparisons(parts: Sequence[dict]) -> dict:
    """Combine per-chunk :func:`compare_outputs` results into one (maxima and sums)."""
    parts = [p for p in parts if p]
    if not parts:
        return {}
    out = dict(parts[0])
    for p in parts[1:]:
        out["orders"] += p["orders"]
        for key in ("levels_used_mismatch", "max_last_level_mismatch", "nan_mismatch"):
            out[key] += p[key]
        for key in ("max_avg_price_tick_err", "max_slip_bps_rel_err", "max_fill_frac_err"):
            out[key] = max(out[key], p[key])
        out["avg_price_within_tol"] = bool(out["avg_price_within_tol"] and p["avg_price_within_tol"])
        out["exact_ticks"] = bool(out["exact_ticks"] and p["exact_ticks"])
        out["passed"] = bool(out["passed"] and p["passed"])
    return out


# --------------------------------------------------------------------------- #
# Dashboard surface
# --------------------------------------------------------------------------- #


def minute_buckets(snap_ts: np.ndarray, max_buckets: int = MAX_SURFACE_BUCKETS) -> Tuple[np.ndarray, int, int]:
    """Bucket snapshots by minute, merging ``factor`` minutes per bucket when needed.

    Returns ``(bucket_ids, factor, first_minute)``: with ``span`` distinct
    minute slots in the capture window, ``factor = ceil(span / max_buckets)``
    so the number of buckets never exceeds ``max_buckets``.
    """
    minutes = np.asarray(snap_ts, dtype=np.int64) // NS_PER_MINUTE
    m0 = int(minutes.min())
    span = int(minutes.max()) - m0 + 1
    factor = max(1, -(-span // max(1, int(max_buckets))))
    return ((minutes - m0) // factor).astype(np.int64), factor, m0


def _bucket_median(block: np.ndarray) -> np.ndarray:
    if np.isnan(block).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(block, axis=0)
    return np.median(block, axis=0)


def build_surface(slip_bps: np.ndarray, snap_ts: np.ndarray, snap_product: np.ndarray,
                  products: Sequence[str], sizes_usd: Sequence[float] = SIZES_USD,
                  max_buckets: int = MAX_SURFACE_BUCKETS, max_bytes: int = MAX_SURFACE_BYTES,
                  decimals: int = 3) -> Tuple[dict, str]:
    """Per product and side, per-minute median slippage (bps) for every size.

    ``slip_bps`` is ``(n, 2, n_sizes)``. Returns ``(surface, json_text)`` where
    the JSON is compact and under ``max_bytes``; if it is not, the bucket cap
    is halved (merging more minutes per bucket) until it is.
    """
    slip = np.asarray(slip_bps)
    snap_product = np.asarray(snap_product)
    cap = int(max_buckets)
    while True:
        bucket, factor, m0 = minute_buckets(snap_ts, cap)
        surface: dict = {
            "sizes_usd": [round(float(s), decimals) for s in sizes_usd],
            "bucket_minutes": int(factor),
            "sides": list(SIDES),
            "products": {},
        }
        for pi, name in enumerate(products):
            rows = np.flatnonzero(snap_product == pi)
            if rows.size == 0:
                continue
            order = np.argsort(bucket[rows], kind="stable")
            rows = rows[order]
            uniq, starts = np.unique(bucket[rows], return_index=True)
            groups = np.split(rows, starts[1:])
            minutes = [int(m0 + int(u) * factor) for u in uniq]
            per_side = {}
            for si, side in enumerate(SIDES):
                bps = []
                for g in groups:
                    med = _bucket_median(slip[g, si, :].astype(np.float64))
                    bps.append([round(float(v), decimals) if np.isfinite(v) else None for v in med])
                per_side[side] = {"minutes": minutes, "bps": bps}
            surface["products"][name] = per_side
        text = json.dumps(surface, separators=(",", ":"), allow_nan=False)
        if len(text.encode("utf-8")) <= max_bytes or cap <= 1:
            return surface, text
        cap //= 2
