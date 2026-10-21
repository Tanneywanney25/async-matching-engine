"""Write a synthetic archive with the exact schema of the real one.

    python scripts/synth_archive.py [--out data/synth] [--snapshots 200000] [--seed 1]

Prices follow a per-product random walk in ticks around realistic levels, book
levels are strictly ordered with lognormal sizes, timestamps advance by random
10 ms to 200 ms steps per product, and trades are scattered through the window.
The sidecar says ``"source": "synthetic"`` so the doc renderer refuses it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exchange_engine.archive import LEVELS, PRODUCTS, ArchiveWriter, validate  # noqa: E402

START_TICKS = {"BTC-USD": 11_000_000, "ETH-USD": 390_000}  # $110,000.00 and $3,900.00


def generate(out_dir: Path, n_snapshots: int, n_trades: int, seed: int,
             products: List[str], levels: int = LEVELS) -> dict:
    rng = np.random.default_rng(seed)
    writer = ArchiveWriter(out_dir, source="synthetic", products=products, levels=levels)
    n_products = len(products)
    base_ns = 1_760_000_000_000_000_000
    # Interleave products: product of snapshot i is i % n_products; per-product time steps.
    product = (np.arange(n_snapshots) % n_products).astype(np.uint8)
    step_ns = rng.integers(10_000_000, 200_000_000, size=n_snapshots, dtype=np.int64)
    ts = base_ns + np.cumsum(step_ns)
    chunk = 65_536
    mids = {p: START_TICKS.get(p, 1_000_000) for p in products}
    for start in range(0, n_snapshots, chunk):
        stop = min(start + chunk, n_snapshots)
        n = stop - start
        prod = product[start:stop]
        mid = np.empty(n, dtype=np.int64)
        for idx, p in enumerate(products):
            m = prod == idx
            k = int(m.sum())
            if k == 0:
                continue
            walk = np.cumsum(rng.integers(-3, 4, size=k, dtype=np.int64))
            vals = mids[p] + walk
            mid[m] = vals
            mids[p] = int(vals[-1])
        half_spread = rng.integers(1, 4, size=n, dtype=np.int64)  # 1..3 ticks
        best_bid = mid - half_spread
        best_ask = mid + half_spread + (rng.random(n) < 0.3)  # occasionally even spread
        gaps_b = rng.integers(1, 3, size=(n, levels), dtype=np.int64)
        gaps_a = rng.integers(1, 3, size=(n, levels), dtype=np.int64)
        gaps_b[:, 0] = 0
        gaps_a[:, 0] = 0
        bid_px = best_bid[:, None] - np.cumsum(gaps_b, axis=1)
        ask_px = best_ask[:, None] + np.cumsum(gaps_a, axis=1)
        bid_sz = rng.lognormal(mean=-1.5, sigma=1.0, size=(n, levels)).astype(np.float32)
        ask_sz = rng.lognormal(mean=-1.5, sigma=1.0, size=(n, levels)).astype(np.float32)
        bid_sz = np.maximum(bid_sz, np.float32(1e-5))
        ask_sz = np.maximum(ask_sz, np.float32(1e-5))
        writer.add_snapshot_batch(ts[start:stop], prod, bid_px.astype(np.int32), bid_sz,
                                  ask_px.astype(np.int32), ask_sz)
    # Trades: uniformly spread in time, priced near the walk, maker side random.
    t_ts = np.sort(rng.integers(ts[0], ts[-1] + 1, size=n_trades, dtype=np.int64))
    t_prod = rng.integers(0, n_products, size=n_trades).astype(np.uint8)
    t_px = np.empty(n_trades, dtype=np.int64)
    for idx, p in enumerate(products):
        m = t_prod == idx
        k = int(m.sum())
        t_px[m] = START_TICKS.get(p, 1_000_000) + np.cumsum(rng.integers(-2, 3, size=k))
    t_sz = np.maximum(rng.lognormal(mean=-3.0, sigma=1.2, size=n_trades), 1e-5).astype(np.float32)
    t_side = np.where(rng.random(n_trades) < 0.5, 1, -1).astype(np.int8)
    writer.add_trade_batch(t_ts, t_ts - 2_000_000, t_prod, t_px.astype(np.int32), t_sz, t_side)
    sidecar = writer.finish({"generator": "scripts/synth_archive.py", "seed": seed,
                             "gap_count": 0, "connect_count": 1})
    validate(out_dir)
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic archive generator")
    parser.add_argument("--out", default="data/synth")
    parser.add_argument("--snapshots", type=int, default=200_000)
    parser.add_argument("--trades", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS))
    args = parser.parse_args()
    sidecar = generate(Path(args.out), args.snapshots, args.trades, args.seed, args.products)
    print(f"synthetic archive: {sidecar['n_snapshots']:,} snapshots, {sidecar['n_trades']:,} trades, "
          f"{sidecar['bytes'] / 1e6:.1f} MB, hash {sidecar['content_hash'][:16]}... -> {args.out}")


if __name__ == "__main__":
    main()
