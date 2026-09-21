"""Matching-engine simulator and FastAPI application.

:class:`MatchingEngine` owns a private :class:`~exchange_engine.orderbook.OrderBook`
that is entirely separate from the live feed mirror: it lets you submit synthetic
orders and see what they would fill against, while tracking cumulative P&L. The
FastAPI app exposes the simulator plus read-only views of the live feed.

Run the API with::

    uvicorn exchange_engine.engine:app --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional

from .feed import CoinbaseFeed
from .models import Fill, Order, OrderType, Side
from .orderbook import OrderBook

logger = logging.getLogger("exchange_engine.engine")

PRODUCT_IDS = ["BTC-USD", "ETH-USD"]


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


# --------------------------------------------------------------------------- #
# FastAPI application                                                          #
# --------------------------------------------------------------------------- #

try:
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:  # pragma: no cover - FastAPI optional at import time
    FastAPI = None  # type: ignore[assignment]


# Shared singletons used by the API layer.
feed = CoinbaseFeed(PRODUCT_IDS)
engine = MatchingEngine("BTC-USD")


def _levels_to_json(levels: List) -> List[Dict[str, str]]:
    """Serialise a list of BookLevel objects to JSON-friendly dicts."""
    return [{"price": str(lvl.price), "quantity": str(lvl.quantity)} for lvl in levels]


def _parse_side(raw: str) -> Side:
    """Parse a side string (``buy``/``bid`` or ``sell``/``ask``) to :class:`Side`."""
    value = raw.strip().lower()
    if value in ("buy", "bid", "b"):
        return Side.BID
    if value in ("sell", "ask", "offer", "s"):
        return Side.ASK
    raise ValueError(f"invalid side: {raw!r}")


def create_app() -> "FastAPI":
    """Build and configure the FastAPI application."""
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("fastapi is not installed")

    @asynccontextmanager
    async def lifespan(_app: "FastAPI"):
        task = asyncio.create_task(feed.run())
        logger.info("started Coinbase feed background task")
        try:
            yield
        finally:
            await feed.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="exchange-engine",
        description="Real-time order book and matching-engine simulator on Coinbase data",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> Dict[str, object]:
        """Basic health/info endpoint."""
        return {
            "service": "exchange-engine",
            "products": PRODUCT_IDS,
            "last_heartbeat": feed.last_heartbeat,
        }

    @app.post("/order")
    async def post_order(payload: Dict = Body(...)) -> Dict[str, object]:
        """Submit a synthetic order to the internal book and return fills.

        Body: ``{"side": "buy", "price": "64000", "quantity": "0.01",
        "order_type": "limit"}``.
        """
        try:
            side = _parse_side(str(payload["side"]))
            order_type = OrderType(str(payload.get("order_type", "limit")).lower())
            quantity = Decimal(str(payload["quantity"]))
            price = payload.get("price")
            price_dec = Decimal(str(price)) if price is not None else None
        except (KeyError, ValueError, InvalidOperation) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        fills = engine.submit_order(side, quantity, price_dec, order_type)
        mark = feed.metrics(engine.product_id).get("mid_price")
        return {
            "fills": [
                {"price": str(f.price), "quantity": str(f.quantity), "side": f.side.value}
                for f in fills
            ],
            "pnl": engine.pnl_summary(mark),
        }

    @app.get("/book")
    async def get_book(
        product_id: str = Query("BTC-USD"), levels: int = Query(10, ge=1, le=50)
    ) -> Dict[str, object]:
        """Return the top ``levels`` bid/ask levels from the live feed mirror."""
        depth = feed.get_depth(product_id, levels)
        return {
            "product_id": product_id,
            "bids": _levels_to_json(depth["bids"]),
            "asks": _levels_to_json(depth["asks"]),
        }

    @app.get("/trades")
    async def get_trades(
        product_id: str = Query("BTC-USD"), limit: int = Query(50, ge=1, le=500)
    ) -> Dict[str, object]:
        """Return recent trades for ``product_id`` from the feed (newest last)."""
        trades = feed.recent_trades(product_id, limit)
        return {
            "product_id": product_id,
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "price": str(t.price),
                    "size": str(t.size),
                    "side": t.side,
                    "time": t.time,
                }
                for t in trades
            ],
        }

    @app.get("/metrics")
    async def get_metrics(product_id: str = Query("BTC-USD")) -> Dict[str, object]:
        """Return spread, mid price, VWAP and imbalance for ``product_id``."""
        m = feed.metrics(product_id)
        return {
            "product_id": product_id,
            "spread": None if m["spread"] is None else str(m["spread"]),
            "mid_price": None if m["mid_price"] is None else str(m["mid_price"]),
            "vwap": str(m["vwap"]),
            "imbalance": str(m["imbalance"]),
        }

    return app


# Module-level ASGI app for ``uvicorn exchange_engine.engine:app``.
app = create_app() if FastAPI is not None else None
