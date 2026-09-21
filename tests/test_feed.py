"""Tests for the Coinbase feed message parsing and book mirror."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from exchange_engine.feed import CoinbaseFeed

pytestmark = pytest.mark.asyncio


def _msg(obj: dict) -> str:
    return json.dumps(obj)


async def test_snapshot_builds_mirror_and_maps_offer_to_ask() -> None:
    feed = CoinbaseFeed(["BTC-USD"])
    await feed._handle_message(
        _msg(
            {
                "channel": "l2_data",
                "events": [
                    {
                        "type": "snapshot",
                        "product_id": "BTC-USD",
                        "updates": [
                            {"side": "bid", "price_level": "100", "new_quantity": "2"},
                            {"side": "offer", "price_level": "101", "new_quantity": "3"},
                        ],
                    }
                ],
            }
        )
    )
    assert feed.best_bid("BTC-USD") == Decimal("100")
    assert feed.best_ask("BTC-USD") == Decimal("101")  # "offer" mapped to ask


async def test_update_zero_quantity_removes_level() -> None:
    feed = CoinbaseFeed(["BTC-USD"])
    await feed._handle_message(
        _msg(
            {
                "channel": "l2_data",
                "events": [
                    {
                        "type": "snapshot",
                        "product_id": "BTC-USD",
                        "updates": [{"side": "bid", "price_level": "100", "new_quantity": "2"}],
                    }
                ],
            }
        )
    )
    await feed._handle_message(
        _msg(
            {
                "channel": "l2_data",
                "events": [
                    {
                        "type": "update",
                        "product_id": "BTC-USD",
                        "updates": [{"side": "bid", "price_level": "100", "new_quantity": "0"}],
                    }
                ],
            }
        )
    )
    assert feed.best_bid("BTC-USD") is None


async def test_trades_buffer_and_vwap() -> None:
    feed = CoinbaseFeed(["BTC-USD"])
    await feed._handle_message(
        _msg(
            {
                "channel": "market_trades",
                "events": [
                    {
                        "type": "update",
                        "trades": [
                            {"trade_id": "1", "product_id": "BTC-USD", "price": "10",
                             "size": "2", "side": "BUY", "time": "t"},
                            {"trade_id": "2", "product_id": "BTC-USD", "price": "20",
                             "size": "3", "side": "SELL", "time": "t"},
                        ],
                    }
                ],
            }
        )
    )
    assert len(feed.recent_trades("BTC-USD")) == 2
    # (10*2 + 20*3) / 5 = 16
    assert feed.rolling_vwap("BTC-USD") == Decimal("16")


async def test_heartbeat_updates_timestamp() -> None:
    feed = CoinbaseFeed(["BTC-USD"])
    await feed._handle_message(
        _msg({"channel": "heartbeats", "timestamp": "ts", "events": [{"heartbeat_counter": 1}]})
    )
    assert feed.last_heartbeat == "ts"
