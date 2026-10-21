"""NumPy references for the Phase 3 feature kernels (``cuda/features.cu``).

Two entry points over the same level-major arrays:

* :func:`features_reference` -- the verification reference: tick quantities in integer
  arithmetic, every float output computed in float64 from the float32 inputs.
* :func:`features_numpy` -- the timed "vectorized NumPy" baseline: the natural NumPy
  expression of the same features in float32, the way a NumPy user would write it.

Definitions (shared with the CUDA kernels and the C++ baseline):

``spread_ticks``  ask_px[0] - bid_px[0] (int32), 0 when either side is empty;
``spread_usd``    spread_ticks * tick_usd[product];
``depth_bid/ask`` sum of the sizes over all levels (padding levels hold size 0);
``imbalance10``   (B10 - A10) / (B10 + A10) over the top 10 levels, 0 when the total is 0,
                  i.e. :func:`exchange_engine.metrics.book_imbalance` with ``levels=10``;
``microprice_offset_ticks``  microprice - mid in ticks, microprice =
                  (bb * as0 + ba * bs0) / (bs0 + as0), mid = (bb + ba) / 2, which is
                  0.5 * spread * (bs0 - as0) / (bs0 + as0); 0 when a side is empty or
                  both top sizes are 0.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

TOP = 10
OUTPUTS = ("spread_ticks", "spread_usd", "depth_bid", "depth_ask", "imbalance10", "microprice_offset_ticks")
OUTPUT_DTYPES = {"spread_ticks": np.int32, "spread_usd": np.float32, "depth_bid": np.float32,
                 "depth_ask": np.float32, "imbalance10": np.float32, "microprice_offset_ticks": np.float32}
# Floating-point operations per snapshot for the fused computation over 50 levels:
# 100 depth adds, 20 top-10 adds, 3 for the imbalance, 1 for spread_usd, 5 for the microprice.
FLOPS_PER_SNAPSHOT = 129
# Tolerances for float outputs against the float64 reference.
REL_TOL = 1e-5
ABS_TOL = 1e-4


def tick_usd_array(products: Sequence[str], tick_sizes: Mapping[str, str]) -> np.ndarray:
    """float32 tick value in USD per product index."""
    return np.array([float(tick_sizes[p]) for p in products], dtype=np.float32)


def features_reference(bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray,
                       product: np.ndarray, tick_usd: np.ndarray) -> Dict[str, np.ndarray]:
    """Exact-ticks, float64 reference. Arrays are level-major ``(levels, n)``."""
    bb = bid_px[0].astype(np.int64)
    ba = ask_px[0].astype(np.int64)
    both = (bb > 0) & (ba > 0)
    spread = np.where(both, ba - bb, 0)
    bsz = bid_sz.astype(np.float64)
    asz = ask_sz.astype(np.float64)
    b10 = bsz[:TOP].sum(axis=0)
    a10 = asz[:TOP].sum(axis=0)
    tot = b10 + a10
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(tot > 0, (b10 - a10) / np.where(tot > 0, tot, 1.0), 0.0)
        bs0, as0 = bsz[0], asz[0]
        tot0 = bs0 + as0
        micro = np.where(both & (tot0 > 0), 0.5 * spread * (bs0 - as0) / np.where(tot0 > 0, tot0, 1.0), 0.0)
    return {
        "spread_ticks": spread.astype(np.int64),
        "spread_usd": spread * tick_usd.astype(np.float64)[product],
        "depth_bid": bsz.sum(axis=0),
        "depth_ask": asz.sum(axis=0),
        "imbalance10": imb,
        "microprice_offset_ticks": micro,
    }


def features_numpy(bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray,
                   product: np.ndarray, tick_usd: np.ndarray) -> Dict[str, np.ndarray]:
    """The timed vectorized NumPy baseline (float32 throughout, like the kernels)."""
    bb = bid_px[0]
    ba = ask_px[0]
    both = (bb > 0) & (ba > 0)
    spread = np.where(both, ba - bb, 0).astype(np.int32)
    b10 = bid_sz[:TOP].sum(axis=0, dtype=np.float32)
    a10 = ask_sz[:TOP].sum(axis=0, dtype=np.float32)
    tot = b10 + a10
    safe = np.where(tot > 0, tot, np.float32(1))
    imb = np.where(tot > 0, (b10 - a10) / safe, np.float32(0)).astype(np.float32)
    bs0, as0 = bid_sz[0], ask_sz[0]
    tot0 = bs0 + as0
    safe0 = np.where(tot0 > 0, tot0, np.float32(1))
    micro = np.where(both & (tot0 > 0), np.float32(0.5) * spread.astype(np.float32) * ((bs0 - as0) / safe0),
                     np.float32(0)).astype(np.float32)
    return {
        "spread_ticks": spread,
        "spread_usd": spread.astype(np.float32) * tick_usd[product],
        "depth_bid": bid_sz.sum(axis=0, dtype=np.float32),
        "depth_ask": ask_sz.sum(axis=0, dtype=np.float32),
        "imbalance10": imb,
        "microprice_offset_ticks": micro,
    }


def compare(got: Mapping[str, np.ndarray], ref: Mapping[str, np.ndarray], n: int) -> dict:
    """Check ``got`` (first ``n`` entries) against the float64 reference.

    Tick quantities must match exactly; float outputs within ``REL_TOL`` relative or
    ``ABS_TOL`` absolute (elementwise: ``|g - r| <= max(ABS_TOL, REL_TOL * |r|)``).
    """
    out: dict = {"max_abs_err": {}, "max_rel_err": {}, "violations": {}}
    ok = True
    for name in OUTPUTS:
        g = np.asarray(got[name])[:n]
        r = np.asarray(ref[name])[:n]
        if name == "spread_ticks":
            bad = int(np.count_nonzero(g.astype(np.int64) != r.astype(np.int64)))
            out["max_abs_err"][name] = int(np.max(np.abs(g.astype(np.int64) - r.astype(np.int64)))) if n else 0
            out["violations"][name] = bad
            ok = ok and bad == 0
            continue
        g64 = g.astype(np.float64)
        diff = np.abs(g64 - r)
        allowed = np.maximum(ABS_TOL, REL_TOL * np.abs(r))
        bad = int(np.count_nonzero(~(diff <= allowed)))
        out["max_abs_err"][name] = float(diff.max()) if n else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(np.abs(r) > 0, diff / np.abs(r), 0.0)
        out["max_rel_err"][name] = float(rel.max()) if n else 0.0
        out["violations"][name] = bad
        ok = ok and bad == 0
    out["verified"] = bool(ok)
    return out


def bytes_min_io(n: int, levels: int) -> int:
    """Minimum bytes the fused computation must move for ``n`` snapshots.

    Reads: every level's bid and ask size (``2 * levels * 4``), the level-0 bid and ask
    price (``2 * 4``) and the product id (1). Writes: six 4-byte outputs. Prices beyond
    level 0 are never needed by these features, so they are not counted.
    """
    return int(n) * (2 * levels * 4 + 2 * 4 + 1 + 6 * 4)


def pick_best(table: Sequence[Mapping]) -> dict:
    """Lowest median in a launch-configuration sweep table (ties: smaller block, then grid)."""
    rows = [r for r in table if r.get("median_ms") is not None]
    if not rows:
        raise ValueError("empty sweep table")
    return dict(min(rows, key=lambda r: (r["median_ms"], r["block_size"], r["grid"])))
