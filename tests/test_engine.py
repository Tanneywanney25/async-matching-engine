"""Tests for the MatchingEngine simulator and CLI command parser."""

from __future__ import annotations

from decimal import Decimal

import pytest

from exchange_engine.engine import MatchingEngine, parse_command
from exchange_engine.models import OrderType, Side


def test_resting_limit_order_has_no_fills() -> None:
    eng = MatchingEngine("TEST")
    fills = eng.submit_order(Side.BID, Decimal("1"), Decimal("100"))
    assert fills == []
    assert eng.position == Decimal("0")
    assert eng.cash == Decimal("0")


def test_crossing_updates_pnl_accounting() -> None:
    eng = MatchingEngine("TEST")
    eng.submit_order(Side.ASK, Decimal("1"), Decimal("101"))  # rest an offer
    fills = eng.submit_order(Side.BID, Decimal("1"), Decimal("101"))  # lift it
    assert sum(f.quantity for f in fills) == Decimal("1")
    assert eng.position == Decimal("1")
    assert eng.cash == Decimal("-101")
    # Marked at 110 -> cash + position*mark = -101 + 110 = 9
    assert eng.unrealized_pnl(Decimal("110")) == Decimal("9")


def test_market_order_accounting() -> None:
    eng = MatchingEngine("TEST")
    eng.submit_order(Side.BID, Decimal("2"), Decimal("100"))
    fills = eng.submit_order(Side.ASK, Decimal("1"), order_type=OrderType.MARKET)
    assert sum(f.quantity for f in fills) == Decimal("1")
    assert eng.position == Decimal("-1")
    assert eng.cash == Decimal("100")


def test_limit_requires_price() -> None:
    eng = MatchingEngine("TEST")
    with pytest.raises(ValueError):
        eng.submit_order(Side.BID, Decimal("1"), price=None, order_type=OrderType.LIMIT)


def test_pnl_summary_shape() -> None:
    eng = MatchingEngine("TEST")
    summary = eng.pnl_summary()
    assert set(summary) == {"cash", "position", "num_fills", "pnl"}


@pytest.mark.parametrize(
    "line,expected",
    [
        ("BUY 64000 0.01 LIMIT", (Side.BID, OrderType.LIMIT, Decimal("64000"), Decimal("0.01"))),
        ("SELL MARKET 0.5", (Side.ASK, OrderType.MARKET, None, Decimal("0.5"))),
        ("sell 100 2", (Side.ASK, OrderType.LIMIT, Decimal("100"), Decimal("2"))),
    ],
)
def test_parse_command(line: str, expected: tuple) -> None:
    assert parse_command(line) == expected


def test_parse_command_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        parse_command("BUY MARKET")  # market needs exactly one quantity
