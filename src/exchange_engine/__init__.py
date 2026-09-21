"""exchange_engine: a real-time limit order book and matching engine.

The package consumes live Coinbase Advanced Trade WebSocket market data and
maintains an in-memory limit order book with price-time priority matching.
"""

from __future__ import annotations

from .metrics import book_imbalance, rolling_vwap, trade_flow_imbalance
from .models import BookLevel, Fill, Order, OrderStatus, OrderType, Side, Trade
from .orderbook import OrderBook

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Side",
    "OrderType",
    "OrderStatus",
    "Order",
    "Fill",
    "BookLevel",
    "Trade",
    "OrderBook",
    "rolling_vwap",
    "book_imbalance",
    "trade_flow_imbalance",
]
