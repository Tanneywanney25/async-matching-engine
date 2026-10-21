"""Deliberately naive list-based limit order book: the differential-testing oracle.

:class:`ReferenceBook` exposes the same subset of the API as
:class:`exchange_engine.orderbook.OrderBook` (``add_order``, ``match_market_order``,
``cancel_order``, ``best_bid``, ``best_ask``, ``get_depth``) but keeps every resting
order in ONE Python list in arrival order and scans it linearly for the best price.
Ties at a price go to the order that appears first in the list, which is exactly
price-time priority. Fills use the resting (maker) price. Nothing here is clever on
purpose: correctness should be obvious from reading it, so that any disagreement with
the real book found by ``scripts/fuzz_engine.py`` points at the real book (or at the
spec), never at the oracle. It is O(n) per operation and is never used in the live path.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .models import BookLevel, Fill, Order, OrderStatus, OrderType, Side


class ReferenceBook:
    """Linear-scan limit order book with the same semantics as :class:`OrderBook`."""

    def __init__(self, product_id: str = "REF") -> None:
        self.product_id = product_id
        self._resting: List[Order] = []  # every resting order, oldest first; the only state

    # -- internals ---------------------------------------------------------

    def _best(self, side: Side) -> Optional[Order]:
        """Earliest resting order at the best price on ``side`` (``None`` if empty)."""
        best: Optional[Order] = None
        for order in self._resting:  # strict comparisons keep the earliest order at a tie
            if order.side is not side:
                continue
            if best is None:
                best = order
            elif side is Side.BID and order.price > best.price:
                best = order
            elif side is Side.ASK and order.price < best.price:
                best = order
        return best

    def _remove(self, order: Order) -> None:
        """Drop ``order`` from the resting list by identity (never by dataclass equality)."""
        self._resting = [o for o in self._resting if o is not order]

    def _consume(
        self, taker_side: Side, quantity: Decimal, taker_id: int, limit_price: Optional[Decimal]
    ) -> Tuple[List[Fill], Decimal]:
        """Fill ``quantity`` against the opposite side at the maker's price, best first."""
        fills: List[Fill] = []
        remaining = quantity
        while remaining > 0:
            maker = self._best(taker_side.opposite)
            if maker is None:
                break
            if limit_price is not None:
                if taker_side is Side.BID and maker.price > limit_price:
                    break
                if taker_side is Side.ASK and maker.price < limit_price:
                    break
            traded = min(remaining, maker.quantity)
            fills.append(
                Fill(order_id=taker_id, price=maker.price, quantity=traded,
                     timestamp=time.time(), side=taker_side)
            )
            maker.quantity -= traded
            remaining -= traded
            if maker.quantity == 0:
                maker.status = OrderStatus.FILLED
                self._remove(maker)
            else:
                maker.status = OrderStatus.PARTIAL
        return fills, remaining

    # -- public API (same subset and same semantics as OrderBook) -----------

    def add_order(self, order: Order) -> List[Fill]:
        """Match a limit order, rest any remainder, and return the taker's fills."""
        if order.order_type is not OrderType.LIMIT:
            raise ValueError("add_order only accepts LIMIT orders; use match_market_order")
        if order.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {order.quantity}")
        if order.price <= 0:
            raise ValueError(f"limit price must be positive, got {order.price}")
        if any(o.id == order.id for o in self._resting):
            raise ValueError(f"duplicate resting order id: {order.id}")
        submitted = order.quantity
        fills, remaining = self._consume(order.side, order.quantity, order.id, order.price)
        order.quantity = remaining
        if remaining == 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL if remaining < submitted else OrderStatus.OPEN
            self._resting.append(order)
        return fills

    def match_market_order(
        self, side: Side, quantity: Decimal, order_id: Optional[int] = None
    ) -> List[Fill]:
        """Fill up to ``quantity`` at any price; the unfilled remainder is dropped."""
        quantity = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
        if quantity <= 0:
            raise ValueError(f"market order quantity must be positive, got {quantity}")
        taker_id = -1 if order_id is None else order_id
        fills, _remaining = self._consume(side, quantity, taker_id, limit_price=None)
        return fills

    def cancel_order(self, order_id: int) -> bool:
        """Remove the resting order with ``order_id``; ``False`` if nothing is resting under it."""
        for order in self._resting:
            if order.id == order_id:
                self._remove(order)
                order.status = OrderStatus.CANCELLED
                return True
        return False

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Highest resting bid price, or ``None``."""
        best = self._best(Side.BID)
        return None if best is None else best.price

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Lowest resting ask price, or ``None``."""
        best = self._best(Side.ASK)
        return None if best is None else best.price

    def get_depth(self, n_levels: int = 10) -> Dict[str, List[BookLevel]]:
        """Top ``n_levels`` aggregated levels per side: bids descending, asks ascending."""
        depth: Dict[str, List[BookLevel]] = {}
        for name, side, descending in (("bids", Side.BID, True), ("asks", Side.ASK, False)):
            totals: Dict[Decimal, Decimal] = {}
            for order in self._resting:
                if order.side is side:
                    totals[order.price] = totals.get(order.price, Decimal(0)) + order.quantity
            prices = sorted(totals, reverse=descending)[:n_levels]
            depth[name] = [BookLevel(price=p, quantity=totals[p]) for p in prices]
        return depth

    def __len__(self) -> int:
        """Number of resting orders across both sides."""
        return len(self._resting)
