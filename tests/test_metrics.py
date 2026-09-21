"""Tests for the analytics helpers in :mod:`exchange_engine.metrics`."""

from __future__ import annotations

from decimal import Decimal

from exchange_engine.metrics import book_imbalance, rolling_vwap, trade_flow_imbalance
from exchange_engine.models import BookLevel, Trade


def _trade(price: str, size: str, side: str = "BUY") -> Trade:
    return Trade(
        trade_id="t",
        product_id="BTC-USD",
        price=Decimal(price),
        size=Decimal(size),
        side=side,
        time="2024-01-01T00:00:00Z",
    )


def test_vwap_known_values() -> None:
    # (10*2 + 20*3) / (2 + 3) = 80 / 5 = 16
    trades = [_trade("10", "2"), _trade("20", "3")]
    assert rolling_vwap(trades) == Decimal("16")


def test_vwap_empty_is_zero() -> None:
    assert rolling_vwap([]) == Decimal("0")


def test_vwap_respects_window() -> None:
    trades = [_trade("100", "1"), _trade("10", "1"), _trade("20", "1")]
    # window of 2 -> (10 + 20) / 2 = 15
    assert rolling_vwap(trades, window=2) == Decimal("15")


def test_imbalance_symmetric_returns_zero() -> None:
    bids = [BookLevel(Decimal("100"), Decimal("5"))]
    asks = [BookLevel(Decimal("101"), Decimal("5"))]
    assert book_imbalance(bids, asks) == Decimal("0")


def test_imbalance_bid_heavy_positive() -> None:
    bids = [BookLevel(Decimal("100"), Decimal("9"))]
    asks = [BookLevel(Decimal("101"), Decimal("1"))]
    assert book_imbalance(bids, asks) == Decimal("0.8")


def test_imbalance_accepts_tuples() -> None:
    assert book_imbalance([(100, 3)], [(101, 1)]) == Decimal("0.5")


def test_trade_flow_imbalance() -> None:
    # SELL maker -> aggressive buy; BUY maker -> aggressive sell.
    trades = [_trade("10", "3", "SELL"), _trade("10", "1", "BUY")]
    # buy_vol=3, sell_vol=1 -> (3-1)/4 = 0.5
    assert trade_flow_imbalance(trades) == Decimal("0.5")
