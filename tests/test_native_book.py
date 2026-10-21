"""The C++20 core through its pybind11 binding: same semantics as the Python engine.

Skipped where ``exchange_engine._native`` has not been built (``cmake -S native -B build/native
&& cmake --build build/native``), unless ``EE_REQUIRE_NATIVE=1`` (CI sets it), which turns a
missing module into a failure so the binding tests can never silently not run.
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from exchange_engine import native_book
from exchange_engine.models import Order, OrderStatus, Side
from exchange_engine.orderbook import OrderBook

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fuzz_engine  # noqa: E402

if not native_book.available():
    if os.environ.get("EE_REQUIRE_NATIVE") == "1":
        raise ImportError("EE_REQUIRE_NATIVE=1 but exchange_engine._native is not built")
    pytest.skip("exchange_engine._native not built", allow_module_level=True)

from exchange_engine import _native  # noqa: E402
from exchange_engine.native_book import NativeOrderBook  # noqa: E402

D = Decimal


def _orders():
    return [Order(Side.BID, D("100.00"), D("5"), id=1), Order(Side.BID, D("100.00"), D("3"), id=2),
            Order(Side.BID, D("99.99"), D("4"), id=3), Order(Side.ASK, D("99.99"), D("10"), id=4),
            Order(Side.ASK, D("99.99"), D("5"), id=5)]


def test_same_fills_statuses_and_depth_as_the_python_engine() -> None:
    py, cc = OrderBook("X"), NativeOrderBook("X")
    py_orders, cc_orders = _orders(), _orders()
    for a, b in zip(py_orders, cc_orders):
        fa, fb = py.add_order(a), cc.add_order(b)
        assert [(f.order_id, f.side, f.price, f.quantity) for f in fa] == [(f.order_id, f.side, f.price, f.quantity) for f in fb]
        assert [(o.id, o.status, o.quantity) for o in py_orders] == [(o.id, o.status, o.quantity) for o in cc_orders]
        assert (py.best_bid, py.best_ask, len(py)) == (cc.best_bid, cc.best_ask, len(cc))
        assert py.get_depth(5) == cc.get_depth(5)
    assert cc_orders[4].status is OrderStatus.PARTIAL and cc_orders[4].quantity == D("3")
    assert cc.cancel_order(5) and not cc.cancel_order(5) and cc_orders[4].status is OrderStatus.CANCELLED
    assert py.cancel_order(5)
    assert cc.book_imbalance() == py.book_imbalance()


def test_market_orders_and_errors_match() -> None:
    cc = NativeOrderBook()
    cc.add_order(Order(Side.ASK, D("101.00"), D("1"), id=1))
    cc.add_order(Order(Side.ASK, D("105.00"), D("2"), id=2))
    fills = cc.match_market_order(Side.BID, D("10"), order_id=9)
    assert [(f.order_id, f.price, f.quantity) for f in fills] == [(9, D("101.00"), D("1")), (9, D("105.00"), D("2"))]
    assert len(cc) == 0 and cc.best_ask is None
    cc.add_order(Order(Side.BID, D("50.00"), D("1"), id=3))
    for bad in (Order(Side.BID, D("50.00"), D("1"), id=3), Order(Side.BID, D("50.00"), D("0"), id=4),
                Order(Side.BID, D("0"), D("1"), id=5)):
        with pytest.raises(ValueError):
            cc.add_order(bad)
    with pytest.raises(ValueError):
        cc.match_market_order(Side.ASK, D("0"))


def test_decimal_is_converted_only_where_it_is_representable() -> None:
    cc = NativeOrderBook()
    with pytest.raises(ValueError, match="ticks"):
        cc.add_order(Order(Side.BID, D("100.005"), D("1"), id=1))
    with pytest.raises(ValueError, match="lots"):
        cc.add_order(Order(Side.BID, D("100.00"), D("0.000000001"), id=2))
    assert len(cc) == 0  # a rejected order never reaches the core


def test_raw_binding_takes_and_returns_integers() -> None:
    b = _native.Book(1 << 10, 64, False)
    assert b.add_limit(1, 0, 10_000, 5) == (_native.STATUS_OPEN, 5, [])
    status, remaining, fills = b.add_limit(2, 1, 9_999, 7)
    assert (status, remaining, fills) == (_native.STATUS_PARTIAL, 2, [(1, 10_000, 5, 0)])
    assert b.best_ask() == 9_999 and b.best_bid() is None and b.depth(1, 3) == [(9_999, 2)]
    assert b.add_limit_count(3, 0, 10_001, 1) == 1 and len(b) == 1
    with pytest.raises(ValueError):
        b.add_limit(2, 1, 9_999, 7)  # id 2 still rests
    with pytest.raises(ValueError):
        _native.Book(1000, 64, False)  # window must be a power of two


def test_fuzzer_finds_no_divergence_on_2000_streams() -> None:
    result = fuzz_engine.run_streams(2000, base_seed=50_000, book_factory=NativeOrderBook, minimize=False)
    assert result.ok, result.divergences[0].report() if result.divergences else ""


class _DropsLastLevel(NativeOrderBook):
    """A deliberately broken book: reports one level too few in depth."""

    def get_depth(self, n_levels: int = 10):
        depth = super().get_depth(n_levels)
        return {"bids": depth["bids"][:-1], "asks": depth["asks"]}


def test_fuzzer_catches_a_broken_native_book() -> None:
    result = fuzz_engine.run_streams(50, book_factory=_DropsLastLevel, minimize=False, max_divergences=1)
    assert not result.ok
