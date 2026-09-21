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


def test_partial_fill_on_resting_order() -> None:
    book = OrderBook("TEST")
    resting = Order(Side.ASK, D("100"), D("3"))
    book.add_order(resting)
    fills = book.add_order(Order(Side.BID, D("100"), D("1")))
    assert sum(f.quantity for f in fills) == D("1")
    assert resting.status == OrderStatus.PARTIAL
    assert resting.quantity == D("2")  # resting order reduced, not removed
    assert book.best_ask == D("100")


def test_cancel_after_partial_fill() -> None:
    book = OrderBook("TEST")
    resting = Order(Side.ASK, D("100"), D("3"))
    book.add_order(resting)
    book.add_order(Order(Side.BID, D("100"), D("1")))  # partially fills resting
    assert book.cancel_order(resting.id) is True
    assert book.best_ask is None
    assert len(book) == 0


def test_duplicate_resting_order_id_rejected() -> None:
    book = OrderBook("TEST")
    first = Order(Side.BID, D("100"), D("1"))
    book.add_order(first)
    clash = Order(Side.BID, D("99"), D("1"))
    object.__setattr__(clash, "id", first.id)  # force an id collision
    with pytest.raises(ValueError, match="duplicate"):
        book.add_order(clash)


@pytest.mark.parametrize("qty", [D("0"), D("-1"), D("-0.001")])
def test_non_positive_quantity_rejected(qty: Decimal) -> None:
    book = OrderBook("TEST")
    with pytest.raises(ValueError, match="quantity must be positive"):
        book.add_order(Order(Side.BID, D("100"), qty))


@pytest.mark.parametrize("price", [D("0"), D("-100")])
def test_non_positive_limit_price_rejected(price: Decimal) -> None:
    book = OrderBook("TEST")
    with pytest.raises(ValueError, match="price must be positive"):
        book.add_order(Order(Side.BID, price, D("1")))


def test_market_order_zero_quantity_rejected() -> None:
    book = OrderBook("TEST")
    with pytest.raises(ValueError, match="quantity must be positive"):
        book.match_market_order(Side.BID, D("0"))


def test_decimal_precision_preserved_with_string_values() -> None:
    book = OrderBook("TEST")
    book.add_order(Order(Side.ASK, D("100.123456789"), D("0.000000010")))
    fills = book.add_order(Order(Side.BID, D("100.123456789"), D("0.000000010")))
    assert fills[0].price == D("100.123456789")
    assert fills[0].quantity == D("0.000000010")
    assert fills[0].notional == D("100.123456789") * D("0.000000010")


def test_empty_book_views_are_none_and_zero() -> None:
    book = OrderBook("TEST")
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.mid_price is None
    assert book.spread is None
    assert book.book_imbalance() == D("0")
    assert book.match_market_order(Side.BID, D("1")) == []


def test_market_order_stops_when_liquidity_exhausted() -> None:
    book = OrderBook("TEST")
    book.add_order(Order(Side.ASK, D("100"), D("1")))
    fills = book.match_market_order(Side.BID, D("5"))
    # Only 1 unit of liquidity exists; the remainder is dropped (no resting).
    assert sum(f.quantity for f in fills) == D("1")
    assert book.best_ask is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
