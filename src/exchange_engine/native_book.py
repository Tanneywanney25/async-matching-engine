"""The C++20 order book core (``native/``) behind the interface of :class:`~exchange_engine.orderbook.OrderBook`.

``NativeOrderBook`` converts :class:`~decimal.Decimal` prices and quantities to integer ticks and
lots at this boundary and nowhere else, calls the compiled core (``exchange_engine._native``,
built with CMake from ``native/``), and turns what comes back into the same
:class:`~exchange_engine.models.Fill` lists and :class:`~exchange_engine.models.Order` status and
quantity updates the Python engine produces, so ``scripts/fuzz_engine.py --engine native`` can
compare the two operation by operation. The Python engine stays the reference implementation.

The one domain difference is representability: the core stores integers, so a price that is not
a whole number of ticks or a quantity that is not a whole number of lots raises ``ValueError``
(the Python engine accepts any Decimal). The defaults, a one-cent tick and a lot of 1e-8, cover
Coinbase's BTC-USD and ETH-USD increments.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Dict, List, Optional

from .models import BookLevel, Fill, Order, OrderStatus, OrderType, Side

try:  # built by `cmake --build build/native` into this package directory
    from . import _native
except ImportError as exc:  # pragma: no cover - exercised only where the module is not built
    _native = None
    _IMPORT_ERROR: Optional[ImportError] = exc
else:
    _IMPORT_ERROR = None

_SIDE = {Side.BID: 0, Side.ASK: 1}
_STATUS = {0: OrderStatus.OPEN, 1: OrderStatus.PARTIAL, 2: OrderStatus.FILLED}


def available() -> bool:
    """True when the compiled core can be imported."""
    return _native is not None


class NativeOrderBook:
    """An in-memory limit order book for one product, matched by the C++ core."""

    def __init__(self, product_id: str = "SIM", tick: Decimal = Decimal("0.01"), lot: Decimal = Decimal("0.00000001"),
                 window_ticks: int = 1 << 18, order_capacity: int = 4096, prefault: bool = False) -> None:
        if _native is None:
            raise ImportError(f"exchange_engine._native is not built ({_IMPORT_ERROR}); "
                              "run `cmake -S native -B build/native && cmake --build build/native`")
        self.product_id = product_id
        self.tick = Decimal(tick)
        self.lot = Decimal(lot)
        self._book = _native.Book(window_ticks, order_capacity, prefault)
        self._orders: Dict[int, Order] = {}  # resting orders, so later fills can update them

    # -- Decimal <-> integers, at the boundary only --------------------------

    def _ticks(self, price: Decimal) -> int:
        q = price / self.tick
        if q != q.to_integral_value():
            raise ValueError(f"price {price} is not a whole number of {self.tick} ticks")
        return int(q)

    def _lots(self, quantity: Decimal) -> int:
        q = quantity / self.lot
        if q != q.to_integral_value():
            raise ValueError(f"quantity {quantity} is not a whole number of {self.lot} lots")
        return int(q)

    def _price(self, ticks: int) -> Decimal:
        return ticks * self.tick

    def _qty(self, lots: int) -> Decimal:
        return lots * self.lot

    def _fills(self, raw: list, taker_id: int, side: Side) -> List[Fill]:
        """Build the taker's fills and apply each one to the resting maker it hit."""
        fills: List[Fill] = []
        for maker_id, price, lots, maker_left in raw:
            fills.append(Fill(order_id=taker_id, price=self._price(price), quantity=self._qty(lots),
                              timestamp=time.time(), side=side))
            maker = self._orders.get(maker_id)
            if maker is not None:
                maker.quantity = self._qty(maker_left)
                if maker_left == 0:
                    maker.status = OrderStatus.FILLED
                    del self._orders[maker_id]
                else:
                    maker.status = OrderStatus.PARTIAL
        return fills

    # -- the OrderBook interface ---------------------------------------------

    def add_order(self, order: Order) -> List[Fill]:
        """Insert a limit order, matching immediately if it crosses; any remainder rests."""
        if order.order_type is not OrderType.LIMIT:
            raise ValueError("add_order only accepts LIMIT orders; use match_market_order")
        if order.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {order.quantity}")
        if order.price <= 0:
            raise ValueError(f"limit price must be positive, got {order.price}")
        if order.id in self._orders:
            raise ValueError(f"duplicate resting order id: {order.id}")
        status, remaining, raw = self._book.add_limit(order.id, _SIDE[order.side], self._ticks(order.price),
                                                      self._lots(order.quantity))
        fills = self._fills(raw, order.id, order.side)
        order.quantity = self._qty(remaining)
        order.status = _STATUS[status]
        if remaining > 0:
            self._orders[order.id] = order
        return fills

    def match_market_order(self, side: Side, quantity: Decimal, order_id: Optional[int] = None) -> List[Fill]:
        """Match a market order against the opposite side; any unfilled remainder is dropped."""
        quantity = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
        if quantity <= 0:
            raise ValueError(f"market order quantity must be positive, got {quantity}")
        taker_id = order_id if order_id is not None else -1
        return self._fills(self._book.match_market(_SIDE[side], self._lots(quantity)), taker_id, side)

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order; ``False`` when no order with that id is resting."""
        if not self._book.cancel(order_id):
            return False
        order = self._orders.pop(order_id, None)
        if order is not None:
            order.status = OrderStatus.CANCELLED
        return True

    @property
    def best_bid(self) -> Optional[Decimal]:
        p = self._book.best_bid()
        return None if p is None else self._price(p)

    @property
    def best_ask(self) -> Optional[Decimal]:
        p = self._book.best_ask()
        return None if p is None else self._price(p)

    @property
    def mid_price(self) -> Optional[Decimal]:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else (bid + ask) / 2

    @property
    def spread(self) -> Optional[Decimal]:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else ask - bid

    def get_depth(self, n_levels: int = 10) -> Dict[str, List[BookLevel]]:
        """Top ``n_levels`` aggregated levels per side: bids descending, asks ascending."""
        return {"bids": [BookLevel(price=self._price(p), quantity=self._qty(q)) for p, q in self._book.depth(0, n_levels)],
                "asks": [BookLevel(price=self._price(p), quantity=self._qty(q)) for p, q in self._book.depth(1, n_levels)]}

    def book_imbalance(self, n_levels: int = 10) -> Decimal:
        depth = self.get_depth(n_levels)
        bid_qty = sum((lvl.quantity for lvl in depth["bids"]), Decimal(0))
        ask_qty = sum((lvl.quantity for lvl in depth["asks"]), Decimal(0))
        total = bid_qty + ask_qty
        return Decimal(0) if total == 0 else (bid_qty - ask_qty) / total

    def __len__(self) -> int:
        return len(self._book)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NativeOrderBook(product_id={self.product_id!r}, resting={len(self)})"
