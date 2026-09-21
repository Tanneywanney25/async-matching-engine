"""Shared data models and enumerations for the exchange engine.

All monetary values use :class:`decimal.Decimal` rather than ``float`` so that
prices and quantities parsed from the Coinbase feed retain full precision.
"""

from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field
from decimal import Decimal


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


# Monotonic counter used to assign unique ids when a caller does not supply one.
_order_id_counter = itertools.count(1)


def _next_order_id() -> int:
    """Return a process-unique, monotonically increasing order id."""
    return next(_order_id_counter)


@dataclass(slots=True)
class Order:
    """A single order submitted to the matching engine.

    Attributes:
        side: Whether the order is a bid (buy) or ask (sell).
        price: Limit price as a :class:`~decimal.Decimal`. Ignored for market orders.
        quantity: Remaining open quantity of the order.
        id: Unique order identifier (auto-assigned when omitted).
        timestamp: Creation time as a Unix epoch float (used for time priority).
        order_type: LIMIT or MARKET.
        status: Current lifecycle status.
    """

    side: Side
    price: Decimal
    quantity: Decimal
    id: int = field(default_factory=_next_order_id)
    timestamp: float = field(default_factory=time.time)
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.OPEN

    def __post_init__(self) -> None:
        """Coerce numeric inputs to :class:`~decimal.Decimal`."""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))


@dataclass(slots=True, frozen=True)
class Fill:
    """An execution that results from matching two orders."""

    order_id: int
    price: Decimal
    quantity: Decimal
    timestamp: float
    side: Side

    @property
    def notional(self) -> Decimal:
        """Cash value of the fill (price * quantity)."""
        return self.price * self.quantity


@dataclass(slots=True, frozen=True)
class BookLevel:
    """A single aggregated price level in the order book."""

    price: Decimal
    quantity: Decimal
