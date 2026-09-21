"""exchange_engine: a real-time limit order book and matching engine.

The package consumes live Coinbase Advanced Trade WebSocket market data and
maintains an in-memory limit order book with price-time priority matching.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
