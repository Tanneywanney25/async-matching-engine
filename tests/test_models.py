"""Tests for the shared data models."""

from __future__ import annotations

from decimal import Decimal

from exchange_engine.models import Fill, Order, OrderType, Side, Trade


def test_order_coerces_numbers_to_decimal() -> None:
    order = Order(Side.BID, "100.5", "0.25")  # type: ignore[arg-type]
    assert order.price == Decimal("100.5")
    assert order.quantity == Decimal("0.25")
    assert isinstance(order.price, Decimal)


def test_order_ids_are_unique_and_monotonic() -> None:
    a = Order(Side.BID, Decimal("1"), Decimal("1"))
    b = Order(Side.BID, Decimal("1"), Decimal("1"))
    assert b.id > a.id


def test_side_opposite() -> None:
    assert Side.BID.opposite is Side.ASK
    assert Side.ASK.opposite is Side.BID


def test_fill_notional() -> None:
    fill = Fill(order_id=1, price=Decimal("100"), quantity=Decimal("0.5"),
                timestamp=0.0, side=Side.BID)
    assert fill.notional == Decimal("50.0")


def test_trade_from_payload_normalizes_side() -> None:
    trade = Trade.from_payload(
        {"trade_id": "7", "product_id": "BTC-USD", "price": "64150.00",
         "size": "0.001", "side": "buy", "time": "2024-06-21T18:29:13.9Z"}
    )
    assert trade.side == "BUY"
    assert trade.price == Decimal("64150.00")
    assert trade.size == Decimal("0.001")


def test_order_defaults_to_limit_open() -> None:
    order = Order(Side.ASK, Decimal("1"), Decimal("1"))
    assert order.order_type is OrderType.LIMIT
