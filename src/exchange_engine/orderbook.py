"""Limit order book with price-time priority matching.

The book keeps two :class:`sortedcontainers.SortedDict` trees, one per side.
Bids are keyed by the *negated* price so that natural ascending key order yields
descending price order (best bid first); asks are keyed by their natural price.
Every price level stores a :class:`collections.deque` of resting orders to give
FIFO time priority within a level. All tree operations are O(log n).
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Deque, Dict, List, Optional, Tuple

from sortedcontainers import SortedDict

from .models import BookLevel, Order, OrderStatus, OrderType, Side


class OrderBook:
    """An in-memory limit order book for a single product."""

    def __init__(self, product_id: str = "SIM") -> None:
        """Create an empty order book.

        Args:
            product_id: Symbol this book represents (informational only).
        """
        self.product_id = product_id
        # price-key -> FIFO deque of resting orders
        self._bids: "SortedDict[Decimal, Deque[Order]]" = SortedDict()
        self._asks: "SortedDict[Decimal, Deque[Order]]" = SortedDict()
        # actual-price -> aggregate resting quantity (O(1) depth lookups)
        self._bid_qty: Dict[Decimal, Decimal] = {}
        self._ask_qty: Dict[Decimal, Decimal] = {}
        # order id -> (order, price-key, side) for O(1) cancel lookup
        self._orders: Dict[int, Tuple[Order, Decimal, Side]] = {}

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _key(side: Side, price: Decimal) -> Decimal:
        """Return the SortedDict key for ``price`` on ``side``."""
        return -price if side is Side.BID else price

    def _book(
        self, side: Side
    ) -> "Tuple[SortedDict[Decimal, Deque[Order]], Dict[Decimal, Decimal]]":
        """Return the ``(tree, qty_map)`` pair for ``side``."""
        if side is Side.BID:
            return self._bids, self._bid_qty
        return self._asks, self._ask_qty

    def _rest(self, order: Order) -> None:
        """Insert ``order`` as a resting level, preserving FIFO priority."""
        tree, qty_map = self._book(order.side)
        key = self._key(order.side, order.price)
        level = tree.get(key)
        if level is None:
            level = deque()
            tree[key] = level
            qty_map[order.price] = Decimal(0)
        level.append(order)
        qty_map[order.price] += order.quantity
        self._orders[order.id] = (order, key, order.side)

    # -- public API -------------------------------------------------------

    def add_order(self, order: Order) -> List["object"]:
        """Insert a limit order, resting it on the book.

        Matching against a crossing spread is added in a later revision.

        Args:
            order: The order to insert.

        Returns:
            The list of fills produced (currently always empty).
        """
        if order.order_type is not OrderType.LIMIT:
            raise ValueError("add_order only accepts LIMIT orders")
        self._rest(order)
        order.status = OrderStatus.OPEN
        return []

    def __len__(self) -> int:
        """Return the number of resting orders across both sides."""
        return len(self._orders)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OrderBook(product_id={self.product_id!r}, "
            f"bids={len(self._bids)}, asks={len(self._asks)})"
        )
