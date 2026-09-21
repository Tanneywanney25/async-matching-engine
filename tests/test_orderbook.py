"""Tests for the limit order book and matching logic."""

from __future__ import annotations

from decimal import Decimal

import pytest

from exchange_engine.models import Order, OrderStatus, Side
from exchange_engine.orderbook import OrderBook


def D(value: str) -> Decimal:
    """Terse Decimal constructor for readable tests."""
    return Decimal(value)


def test_add_limit_order_rests_at_price_level() -> None:
    book = OrderBook("TEST")
    book.add_order(Order(Side.BID, D("100"), D("1")))
    book.add_order(Order(Side.ASK, D("101"), D("2")))
    assert book.best_bid == D("100")
    assert book.best_ask == D("101")
    assert book.spread == D("1")
    assert book.mid_price == D("100.5")
    depth = book.get_depth(5)
    assert depth["bids"][0].quantity == D("1")
    assert depth["asks"][0].quantity == D("2")


def test_price_time_priority() -> None:
    book = OrderBook("TEST")
    first = Order(Side.ASK, D("101"), D("1"))
    second = Order(Side.ASK, D("101"), D("1"))
    book.add_order(first)
    book.add_order(second)
    # An aggressive buy for one unit must consume the earlier order first.
    fills = book.add_order(Order(Side.BID, D("101"), D("1")))
    assert sum(f.quantity for f in fills) == D("1")
    assert first.status == OrderStatus.FILLED
    assert second.status == OrderStatus.OPEN
    assert book.best_ask == D("101")


def test_crossing_order_triggers_match() -> None:
    book = OrderBook("TEST")
    book.add_order(Order(Side.ASK, D("100"), D("1")))
    fills = book.add_order(Order(Side.BID, D("105"), D("1")))
    assert len(fills) == 1
    assert fills[0].price == D("100")  # filled at the resting (maker) price
    assert book.best_ask is None


def test_partial_fill_rests_remainder() -> None:
    book = OrderBook("TEST")
    book.add_order(Order(Side.ASK, D("100"), D("1")))
    incoming = Order(Side.BID, D("100"), D("3"))
    fills = book.add_order(incoming)
    assert sum(f.quantity for f in fills) == D("1")
    assert incoming.status == OrderStatus.PARTIAL
    assert incoming.quantity == D("2")
    assert book.best_bid == D("100")  # remainder rested as a bid


def test_cancel_removes_and_cleans_level() -> None:
    book = OrderBook("TEST")
    order = Order(Side.BID, D("100"), D("1"))
    book.add_order(order)
    assert book.cancel_order(order.id) is True
    assert book.best_bid is None
    assert len(book) == 0
    assert book.cancel_order(order.id) is False  # already gone


def test_market_order_walks_multiple_levels() -> None:
    book = OrderBook("TEST")
    book.add_order(Order(Side.ASK, D("100"), D("1")))
    book.add_order(Order(Side.ASK, D("101"), D("1")))
    book.add_order(Order(Side.ASK, D("102"), D("1")))
    fills = book.match_market_order(Side.BID, D("2.5"))
    assert sum(f.quantity for f in fills) == D("2.5")
    prices = [f.price for f in fills]
    assert prices == [D("100"), D("101"), D("102")]
    assert book.best_ask == D("102")  # 0.5 remaining at the top level


def test_book_imbalance_symmetric_is_zero() -> None:
    book = OrderBook("TEST")
    book.add_order(Order(Side.BID, D("100"), D("2")))
    book.add_order(Order(Side.ASK, D("101"), D("2")))
    assert book.book_imbalance() == D("0")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
