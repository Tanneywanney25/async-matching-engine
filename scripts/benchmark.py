"""Latency benchmark for the :class:`~exchange_engine.orderbook.OrderBook`.

Generates a large batch of random limit orders around a mid-price and measures
per-operation insertion+match latency, then measures cancel latency. Reports
p50/p95/p99 in microseconds.

Run with::

    python scripts/benchmark.py [--orders 100000] [--cancels 10000]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import List

# Allow running directly from a checkout without installing the package.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exchange_engine.models import Order, Side  # noqa: E402
from exchange_engine.orderbook import OrderBook  # noqa: E402


def _percentile(sorted_us: List[float], pct: float) -> float:
    """Return the ``pct`` percentile from an already-sorted list (nearest-rank)."""
    if not sorted_us:
        return 0.0
    rank = max(0, min(len(sorted_us) - 1, int(round(pct / 100 * len(sorted_us))) - 1))
    return sorted_us[rank]


def _report(name: str, latencies_us: List[float]) -> None:
    """Print p50/p95/p99 latency statistics for a labelled operation."""
    latencies_us.sort()
    p50 = _percentile(latencies_us, 50)
    p95 = _percentile(latencies_us, 95)
    p99 = _percentile(latencies_us, 99)
    mean = sum(latencies_us) / len(latencies_us)
    print(f"\n{name} ({len(latencies_us):,} ops)")
    print(f"  mean : {mean:8.2f} us")
    print(f"  p50  : {p50:8.2f} us")
    print(f"  p95  : {p95:8.2f} us")
    print(f"  p99  : {p99:8.2f} us")


def benchmark_inserts(n_orders: int, mid: float, seed: int) -> "tuple[OrderBook, List[int]]":
    """Insert ``n_orders`` random limit orders, timing each add_order call.

    Returns the populated book and the list of resting order ids (for cancels).
    """
    rng = random.Random(seed)
    book = OrderBook("BENCH")
    latencies_us: List[float] = []
    resting_ids: List[int] = []

    for _ in range(n_orders):
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        price = Decimal(f"{rng.gauss(mid, mid * 0.01):.2f}")
        quantity = Decimal(f"{rng.uniform(0.001, 2.0):.6f}")
        order = Order(side, price, quantity)

        start = time.perf_counter_ns()
        book.add_order(order)
        latencies_us.append((time.perf_counter_ns() - start) / 1000.0)

        if order.quantity > 0:  # order rested (fully or partially)
            resting_ids.append(order.id)

    _report("add_order (insert + match)", latencies_us)
    return book, resting_ids


def main() -> None:
    """Parse arguments and run the insertion and cancel benchmarks."""
    parser = argparse.ArgumentParser(description="OrderBook latency benchmark")
    parser.add_argument("--orders", type=int, default=100_000)
    parser.add_argument("--cancels", type=int, default=10_000)
    parser.add_argument("--mid", type=float, default=64_000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Benchmarking OrderBook with {args.orders:,} orders (mid={args.mid:,.0f})...")
    book, resting_ids = benchmark_inserts(args.orders, args.mid, args.seed)

    rng = random.Random(args.seed + 1)
    rng.shuffle(resting_ids)
    to_cancel = resting_ids[: args.cancels]
    cancel_us: List[float] = []
    for order_id in to_cancel:
        start = time.perf_counter_ns()
        book.cancel_order(order_id)
        cancel_us.append((time.perf_counter_ns() - start) / 1000.0)
    if cancel_us:
        _report("cancel_order", cancel_us)

    print("\nTarget: median add_order under 50 us.")


if __name__ == "__main__":
    main()
