"""Shared data models and enumerations for the exchange engine.

All monetary values use :class:`decimal.Decimal` rather than ``float`` so that
prices and quantities parsed from the Coinbase feed retain full precision.
"""

from __future__ import annotations

import enum


class Side(enum.Enum):
    """The side of an order or resting book level."""

    BID = "bid"
    ASK = "ask"

    @property
    def opposite(self) -> "Side":
        """Return the opposing side (BID <-> ASK)."""
        return Side.ASK if self is Side.BID else Side.BID


class OrderType(enum.Enum):
    """Supported order types."""

    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(enum.Enum):
    """Lifecycle status of an order in the book."""

    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
