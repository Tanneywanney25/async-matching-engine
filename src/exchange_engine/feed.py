"""Coinbase Advanced Trade WebSocket consumer.

Connects to the public (no-auth) endpoint ``wss://advanced-trade-ws.coinbase.com``
and subscribes to the ``level2``, ``market_trades`` and ``heartbeats`` channels
(one subscribe message per channel, as the API requires). Incoming messages are
parsed and routed to maintain a local mirror of each product's order book and a
rolling buffer of recent trades.

Only the ``websockets`` library is used for transport; there is no dependency on
any Coinbase SDK.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from decimal import Decimal
from typing import Awaitable, Callable, Deque, Dict, List, Optional

import websockets
from sortedcontainers import SortedDict

from .models import BookLevel, Trade

logger = logging.getLogger("exchange_engine.feed")

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
CHANNELS = ("level2", "market_trades", "heartbeats")

EventCallback = Callable[[dict], Optional[Awaitable[None]]]


class ProductBook:
    """A one-sided-per-tree price mirror of a single product's order book.

    Bids are keyed by negated price so ascending key order is descending price.
    Values are aggregate quantities (Decimals) exactly as reported by the feed.
    """

    def __init__(self) -> None:
        self.bids: "SortedDict[Decimal, Decimal]" = SortedDict()
        self.asks: "SortedDict[Decimal, Decimal]" = SortedDict()

    def clear(self) -> None:
        """Drop all levels (used when a fresh snapshot arrives)."""
        self.bids.clear()
        self.asks.clear()

    def set_level(self, side: str, price: Decimal, quantity: Decimal) -> None:
        """Apply one level update; a ``quantity`` of 0 removes the level.

        Args:
            side: Coinbase side string, ``"bid"`` or ``"offer"``.
            price: Price level.
            quantity: Absolute (not delta) quantity at that price level.
        """
        tree = self.bids if side == "bid" else self.asks
        key = -price if side == "bid" else price
        if quantity == 0:
            tree.pop(key, None)
        else:
            tree[key] = quantity

    def depth(self, n_levels: int = 10) -> Dict[str, List[BookLevel]]:
        """Return the top ``n_levels`` per side as BookLevel lists."""
        bids: List[BookLevel] = []
        for key, qty in self.bids.items():
            if len(bids) >= n_levels:
                break
            bids.append(BookLevel(price=-key, quantity=qty))
        asks: List[BookLevel] = []
        for key, qty in self.asks.items():
            if len(asks) >= n_levels:
                break
            asks.append(BookLevel(price=key, quantity=qty))
        return {"bids": bids, "asks": asks}

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Highest bid price in the mirror, or ``None``."""
        return -self.bids.peekitem(0)[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Lowest ask price in the mirror, or ``None``."""
        return self.asks.peekitem(0)[0] if self.asks else None


class CoinbaseFeed:
    """Async consumer of the Coinbase Advanced Trade WebSocket feed."""

    def __init__(
        self,
        product_ids: List[str],
        url: str = COINBASE_WS_URL,
        on_event: Optional[EventCallback] = None,
        max_trades: int = 500,
    ) -> None:
        """Configure the feed.

        Args:
            product_ids: Products to subscribe to, e.g. ``["BTC-USD", "ETH-USD"]``.
            url: WebSocket endpoint (overridable for testing).
            on_event: Optional coroutine/callable invoked for each parsed event.
            max_trades: Cap on the per-product recent-trades buffer.
        """
        self.product_ids = list(product_ids)
        self.url = url
        self.on_event = on_event
        self.max_trades = max_trades

        self.books: Dict[str, ProductBook] = {pid: ProductBook() for pid in self.product_ids}
        self.trades: Dict[str, Deque[Trade]] = {
            pid: deque(maxlen=max_trades) for pid in self.product_ids
        }
        self.events: "asyncio.Queue[dict]" = asyncio.Queue()

        self.last_heartbeat: Optional[str] = None
        self._heartbeat_counter: Optional[int] = None
        self._running = False

    def _subscribe_messages(self) -> List[dict]:
        """Build the per-channel subscribe payloads (one channel each)."""
        messages: List[dict] = []
        for channel in ("level2", "market_trades"):
            messages.append(
                {"type": "subscribe", "product_ids": self.product_ids, "channel": channel}
            )
        # Heartbeats keep the socket alive and take no product_ids.
        messages.append({"type": "subscribe", "channel": "heartbeats"})
        return messages

    async def _subscribe(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """Send each subscribe message; must happen within 5s of connecting."""
        for message in self._subscribe_messages():
            await ws.send(json.dumps(message))
            logger.debug("sent subscribe: %s", message.get("channel"))

    async def _dispatch(self, event: dict) -> None:
        """Push an event to the queue and optional callback."""
        await self.events.put(event)
        if self.on_event is not None:
            result = self.on_event(event)
            if asyncio.iscoroutine(result):
                await result

    async def _handle_message(self, raw: str) -> None:
        """Parse one raw frame and route it by channel."""
        msg = json.loads(raw)
        channel = msg.get("channel")
        if channel == "l2_data":
            await self._handle_l2(msg)
        elif channel == "market_trades":
            await self._handle_trades(msg)
        elif channel == "heartbeats":
            await self._handle_heartbeat(msg)
        # "subscriptions" and other control frames are ignored.

    async def _handle_l2(self, msg: dict) -> None:
        """Placeholder; implemented in a later revision."""

    async def _handle_trades(self, msg: dict) -> None:
        """Placeholder; implemented in a later revision."""

    async def _handle_heartbeat(self, msg: dict) -> None:
        """Placeholder; implemented in a later revision."""

    async def run(self) -> None:
        """Connect and consume forever, reconnecting with exponential backoff.

        Backoff starts at 2 seconds and doubles up to a 30 second cap; it resets
        to 2 seconds after each successful connection.
        """
        self._running = True
        backoff = 2
        while self._running:
            try:
                async with websockets.connect(self.url, ping_interval=None) as ws:
                    logger.info("connected to %s", self.url)
                    await self._subscribe(ws)
                    backoff = 2  # reset after a healthy connect
                    async for raw in ws:
                        await self._handle_message(raw)
            except asyncio.CancelledError:
                self._running = False
                raise
            except Exception as exc:  # noqa: BLE001 - resilient reconnect loop
                logger.warning("feed disconnected: %s; reconnecting in %ss", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def stop(self) -> None:
        """Signal the run loop to exit after the current iteration."""
        self._running = False
