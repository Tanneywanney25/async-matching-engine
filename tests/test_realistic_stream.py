"""bench/realistic_stream.py turns recorded snapshots and trades into a replayable operation stream.

The archive here is a hand-made fixture of a few snapshots and trades, not recorded data.
"""

from __future__ import annotations

import struct
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import benchmark_native as bn  # noqa: E402
from bench.realistic_stream import derive  # noqa: E402
from exchange_engine.models import Order, Side  # noqa: E402
from exchange_engine.orderbook import OrderBook  # noqa: E402


def fixture_archive() -> dict:
    """Two levels per side; BTC-USD (product 0) changes over three snapshots with trades between."""
    #             snapshot 0        snapshot 1        snapshot 2        other product
    bid_px = np.array([[100, 101, 101, 50], [99, 100, 99, 49]], dtype=np.int64)
    bid_sz = np.array([[1.0, 0.5, 0.25, 9.0], [2.0, 1.0, 3.0, 9.0]])
    ask_px = np.array([[102, 103, 102, 51], [103, 104, 103, 52]], dtype=np.int64)
    ask_sz = np.array([[1.5, 2.0, 0.75, 9.0], [1.0, 1.0, 0.0, 9.0]])
    return {
        "meta": {"products": ["BTC-USD", "ETH-USD"]},
        "snap_product": np.array([0, 0, 0, 1]), "snap_ts": np.array([0, 2_000_000_000, 4_000_000_000, 1], dtype=np.int64),
        "bid_px": bid_px, "bid_sz": bid_sz, "ask_px": ask_px, "ask_sz": ask_sz,
        # a maker BUY (side 1) is a seller hitting the bid; a maker SELL (side 0) a buyer lifting the ask
        "trade_product": np.array([0, 0, 1]), "trade_ts": np.array([1_000_000_000, 3_000_000_000, 2], dtype=np.int64),
        "trade_sz": np.array([0.25, 0.5, 1.0]), "trade_side": np.array([1, 0, 1]),
    }


def replay(ops, book) -> list:
    """Every operation through ``book``; returns its fills (or the cancel result) per operation."""
    out = []
    for op in ops:
        if op[0] == "limit":
            side = Side.BID if op[2] == "bid" else Side.ASK
            out.append([(f.order_id, f.price, f.quantity) for f in book.add_order(Order(side, op[3] * bn.TICK, op[4] * bn.LOT, id=op[1]))])
        elif op[0] == "market":
            side = Side.BID if op[2] == "bid" else Side.ASK
            out.append([(f.order_id, f.price, f.quantity) for f in book.match_market_order(side, op[3] * bn.LOT, op[1])])
        else:
            out.append(book.cancel_order(op[1]))
    return out


def test_first_snapshot_becomes_resting_limits() -> None:
    stream = derive(fixture_archive())
    first = stream["ops"][:4]
    assert all(op[0] == "limit" for op in first)
    assert sorted((op[2], op[3], op[4]) for op in first) == [
        ("ask", 102, 150_000_000), ("ask", 103, 100_000_000), ("bid", 99, 200_000_000), ("bid", 100, 100_000_000)]


def test_trades_become_aggressor_side_market_orders() -> None:
    markets = [op for op in derive(fixture_archive())["ops"] if op[0] == "market"]
    assert [(op[2], op[3]) for op in markets] == [("ask", 25_000_000), ("bid", 50_000_000)]


def test_every_cancel_names_a_resting_order_and_the_book_ends_at_the_last_snapshot() -> None:
    stream = derive(fixture_archive())
    book = OrderBook("CHECK")
    results = replay(stream["ops"], book)
    cancels = [r for op, r in zip(stream["ops"], results) if op[0] == "cancel"]
    assert cancels and all(cancels)
    assert stream["counts"]["cancel"] == len(cancels)
    bids = {p: q for p, q in book._bid_qty.items()}  # noqa: SLF001
    asks = {p: q for p, q in book._ask_qty.items()}  # noqa: SLF001
    assert bids == {Decimal("1.01"): Decimal("0.25"), Decimal("0.99"): Decimal("3")}
    # the empty second ask level narrows the reconciled range to 1.02, so the deeper asks stay resting
    assert asks == {Decimal("1.02"): Decimal("0.75"), Decimal("1.03"): Decimal("1.5"), Decimal("1.04"): Decimal("1")}
    assert stream["snapshots_used"] == 3 and stream["trades_used"] == 2
    assert stream["limit_orders_matched"] == 0 and stream["limit_fills"] == 0
    assert stream["span_seconds"] == pytest.approx(4.0)
    assert "first 0.1 minutes" in stream["description"]


def test_a_limit_order_crossing_a_stale_level_is_counted() -> None:
    arch = fixture_archive()
    # snapshot 2 shows no asks, so that side is not reconciled and the resting asks stay;
    # its new best bid at 1.03 then crosses the resting ask at 1.03 on arrival
    arch["ask_sz"][:, 2] = 0.0
    arch["bid_px"][0, 2], arch["bid_sz"][0, 2] = 103, 0.25
    stream = derive(arch)
    assert stream["limit_orders_matched"] == 1 and stream["limit_fills"] == 1


def test_max_ops_stops_after_the_event_that_reaches_it() -> None:
    full = derive(fixture_archive())
    cut = derive(fixture_archive(), max_ops=3)
    assert cut["snapshots_used"] == 1 and len(cut["ops"]) == 4
    assert cut["ops"] == full["ops"][:4]


def test_replay_is_identical_through_the_native_book() -> None:
    from exchange_engine import native_book

    if not native_book.available():
        pytest.skip("the C++ core is not built")
    ops = derive(fixture_archive())["ops"]
    assert replay(ops, native_book.NativeOrderBook("N")) == replay(ops, OrderBook("P"))
    py = bn.time_ops_book(ops, OrderBook("P"))
    nat = bn.time_ops_binding(ops)
    assert (py["fills"], py["resting_at_end"], py["cancels_found"]) == (nat["fills"], nat["resting_at_end"], nat["cancels_found"])
    assert {k: len(v) for k, v in py["us"].items()} == {k: len(v) for k, v in nat["us"].items()}


def test_ops_file_layout(tmp_path: Path) -> None:
    ops = [("limit", 1, "bid", 100, 5), ("market", 2, "ask", 3), ("cancel", 1)]
    path = tmp_path / "ops.bin"
    bn.write_ops(path, ops)
    data = path.read_bytes()
    assert data[:8] == b"EEOPSV02" and struct.unpack("<QQ", data[8:24]) == (3, 0)
    records = [struct.unpack("<BB6xqqq", data[24 + 32 * i:56 + 32 * i]) for i in range(3)]
    assert records == [(0, 0, 1, 100, 5), (1, 1, 2, 0, 3), (2, 0, 1, 0, 0)]
    assert len(data) == 24 + 32 * 3
    digest = bn.file_digest(path)
    assert digest["bytes"] == len(data) and len(digest["sha256"]) == 64
