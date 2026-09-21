"""Matching-engine simulator and FastAPI application.

:class:`MatchingEngine` owns a private :class:`~exchange_engine.orderbook.OrderBook`
that is entirely separate from the live feed mirror: it lets you submit synthetic
orders and see what they would fill against, while tracking cumulative P&L. The
FastAPI app exposes the simulator plus read-only views of the live feed.

Run the API with::

    uvicorn exchange_engine.engine:app --port 8000
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional

from .models import Fill, Order, OrderType, Side
from .orderbook import OrderBook


class MatchingEngine:
    """A single-book order simulator with cumulative P&L accounting."""

    def __init__(self, product_id: str = "BTC-USD") -> None:
        """Create an engine backed by an empty internal order book."""
        self.product_id = product_id
        self.book = OrderBook(product_id)
        self.fills: List[Fill] = []
        # Cash flow (signed) and net base position from all synthetic fills.
        self.cash = Decimal(0)
        self.position = Decimal(0)

    def _account_for(self, fills: List[Fill]) -> None:
        """Update cash and position from a batch of taker fills."""
        for fill in fills:
            if fill.side is Side.BID:  # bought base, paid cash
                self.cash -= fill.price * fill.quantity
                self.position += fill.quantity
            else:  # sold base, received cash
                self.cash += fill.price * fill.quantity
                self.position -= fill.quantity
        self.fills.extend(fills)

    def submit_order(
        self,
        side: Side,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        order_type: OrderType = OrderType.LIMIT,
    ) -> List[Fill]:
        """Submit a synthetic order to the internal book and record its fills.

        Args:
            side: BID (buy) or ASK (sell).
            quantity: Order quantity.
            price: Limit price (required for LIMIT orders, ignored for MARKET).
            order_type: LIMIT or MARKET.

        Returns:
            The fills the order received.
        """
        quantity = Decimal(str(quantity))
        if order_type is OrderType.MARKET:
            fills = self.book.match_market_order(side, quantity)
        else:
            if price is None:
                raise ValueError("LIMIT orders require a price")
            order = Order(side=side, price=Decimal(str(price)), quantity=quantity)
            fills = self.book.add_order(order)
        self._account_for(fills)
        return fills

    def unrealized_pnl(self, mark_price: Optional[Decimal]) -> Decimal:
        """Mark-to-market P&L: ``cash + position * mark_price``.

        Args:
            mark_price: Current reference price; when ``None`` only realized cash
                flow is returned.
        """
        if mark_price is None:
            return self.cash
        return self.cash + self.position * mark_price

    def pnl_summary(self, mark_price: Optional[Decimal] = None) -> Dict[str, object]:
        """Return a serialisable snapshot of the engine's P&L state."""
        return {
            "cash": str(self.cash),
            "position": str(self.position),
            "num_fills": len(self.fills),
            "pnl": str(self.unrealized_pnl(mark_price)),
        }
