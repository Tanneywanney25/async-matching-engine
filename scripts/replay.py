"""Offline replay: drive a CoinbaseFeed with synthetic frames (no network).

Generates a snapshot followed by a random walk of level2 updates and trades,
then prints the resulting book and metrics. Useful for demoing the pipeline or
sanity-checking parsing without a live connection.

Run with::

    python scripts/replay.py [--steps 200]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exchange_engine.feed import CoinbaseFeed  # noqa: E402

PRODUCT = "BTC-USD"


def _snapshot(mid: float) -> str:
    updates = []
    for i in range(1, 11):
        updates.append({"side": "bid", "price_level": f"{mid - i:.2f}", "new_quantity": f"{random.uniform(0.1, 2):.4f}"})
        updates.append({"side": "offer", "price_level": f"{mid + i:.2f}", "new_quantity": f"{random.uniform(0.1, 2):.4f}"})
    return json.dumps(
        {"channel": "l2_data", "events": [{"type": "snapshot", "product_id": PRODUCT, "updates": updates}]}
    )


def _update(mid: float) -> str:
    side = random.choice(["bid", "offer"])
    offset = random.randint(1, 10)
    price = mid - offset if side == "bid" else mid + offset
    qty = random.choice([0, round(random.uniform(0.1, 2), 4)])
    return json.dumps(
        {
            "channel": "l2_data",
            "events": [
                {"type": "update", "product_id": PRODUCT,
                 "updates": [{"side": side, "price_level": f"{price:.2f}", "new_quantity": str(qty)}]}
            ],
        }
    )


def _trade(mid: float) -> str:
    return json.dumps(
        {
            "channel": "market_trades",
            "events": [
                {"type": "update", "trades": [
                    {"trade_id": str(random.randint(1, 1_000_000)), "product_id": PRODUCT,
                     "price": f"{mid + random.uniform(-1, 1):.2f}", "size": f"{random.uniform(0.001, 0.5):.6f}",
                     "side": random.choice(["BUY", "SELL"]), "time": "2024-01-01T00:00:00Z"}]}
            ],
        }
    )


async def run(steps: int, seed: int) -> None:
    random.seed(seed)
    feed = CoinbaseFeed([PRODUCT])
    mid = 64_000.0
    await feed.apply(_snapshot(mid))
    for step in range(steps):
        mid += random.uniform(-1.5, 1.5)
        # Periodically re-snapshot so drifting synthetic levels never cross.
        if step % 20 == 19:
            await feed.apply(_snapshot(mid))
        else:
            await feed.apply(_update(mid))
        if random.random() < 0.4:
            await feed.apply(_trade(mid))

    depth = feed.get_depth(PRODUCT, 5)
    print(f"Replayed {steps} steps for {PRODUCT}")
    print("Best bid:", feed.best_bid(PRODUCT), "Best ask:", feed.best_ask(PRODUCT))
    print("Top bids:", [(str(l.price), str(l.quantity)) for l in depth["bids"]])
    print("Top asks:", [(str(l.price), str(l.quantity)) for l in depth["asks"]])
    print("Metrics:", {k: str(v) for k, v in feed.metrics(PRODUCT).items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline synthetic feed replay")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(run(args.steps, args.seed))


if __name__ == "__main__":
    main()
