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

import time

from .models import BookLevel, Fill, Order, OrderStatus, OrderType, Side


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

    def _consume(
        self,
        taker_side: Side,
        quantity: Decimal,
        taker_id: int,
        limit_price: Optional[Decimal],
    ) -> Tuple[List[Fill], Decimal]:
        """Walk the opposite side, filling ``quantity`` at the best prices.

        Args:
            taker_side: Side of the incoming (aggressing) order.
            quantity: Quantity to fill.
            taker_id: Id recorded on the resulting taker fills.
            limit_price: Price limit for the taker, or ``None`` for a market order.

        Returns:
            A ``(fills, remaining_quantity)`` tuple. ``remaining_quantity`` is the
            portion that could not be matched at acceptable prices.
        """
        opp_tree, opp_qty = self._book(taker_side.opposite)
        fills: List[Fill] = []
        remaining = quantity

        while remaining > 0 and opp_tree:
            key, level = opp_tree.peekitem(0)  # best opposing level
            level_price = -key if taker_side.opposite is Side.BID else key
            # Respect the taker's limit price (market orders pass through).
            if limit_price is not None:
                if taker_side is Side.BID and level_price > limit_price:
                    break
                if taker_side is Side.ASK and level_price < limit_price:
                    break

            while level and remaining > 0:
                resting = level[0]
                traded = min(remaining, resting.quantity)
                fills.append(
                    Fill(
                        order_id=taker_id,
                        price=level_price,
                        quantity=traded,
                        timestamp=time.time(),
                        side=taker_side,
                    )
                )
                resting.quantity -= traded
                remaining -= traded
                opp_qty[level_price] -= traded
                if resting.quantity == 0:
                    resting.status = OrderStatus.FILLED
                    level.popleft()
                    self._orders.pop(resting.id, None)
                else:
                    resting.status = OrderStatus.PARTIAL

            if not level:
                del opp_tree[key]
                opp_qty.pop(level_price, None)

        return fills, remaining

    # -- public API -------------------------------------------------------

    def add_order(self, order: Order) -> List[Fill]:
        """Insert a limit order, matching immediately if it crosses the spread.

        Any residual quantity that does not cross is rested on the book with
        price-time priority.

        Args:
            order: The limit order to insert.

        Returns:
            The list of fills the incoming order received (may be empty).
        """
        if order.order_type is not OrderType.LIMIT:
            raise ValueError("add_order only accepts LIMIT orders; use match_market_order")

        original_qty = order.quantity
        fills, remaining = self._consume(order.side, order.quantity, order.id, order.price)
        order.quantity = remaining

        if remaining == 0:
            order.status = OrderStatus.FILLED
        elif remaining < original_qty:
            order.status = OrderStatus.PARTIAL
            self._rest(order)
        else:
            order.status = OrderStatus.OPEN
            self._rest(order)
        return fills

    def match_market_order(
        self, side: Side, quantity: Decimal, order_id: Optional[int] = None
    ) -> List[Fill]:
        """Match a market order against the opposite side of the book.

        Args:
            side: Side of the aggressing market order.
            quantity: Quantity to execute.
            order_id: Optional id to stamp on the resulting fills.

        Returns:
            The fills produced while walking the book. Any unfilled remainder is
            simply dropped (market orders do not rest).
        """
        quantity = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
        taker_id = order_id if order_id is not None else -1
        fills, _remaining = self._consume(side, quantity, taker_id, limit_price=None)
        return fills

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order and clean up its price level if now empty.

        Args:
            order_id: Id of the order to remove.

        Returns:
            ``True`` if an order was found and cancelled, ``False`` otherwise.
        """
        entry = self._orders.pop(order_id, None)
        if entry is None:
            return False
        order, key, side = entry
        tree, qty_map = self._book(side)
        level = tree.get(key)
        if level is not None:
            try:
                level.remove(order)
            except ValueError:  # pragma: no cover - defensive
                pass
            qty_map[order.price] -= order.quantity
            if not level:
                del tree[key]
                qty_map.pop(order.price, None)
        order.status = OrderStatus.CANCELLED
        return True

    def __len__(self) -> int:
        """Return the number of resting orders across both sides."""
        return len(self._orders)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OrderBook(product_id={self.product_id!r}, "
            f"bids={len(self._bids)}, asks={len(self._asks)})"
        )
