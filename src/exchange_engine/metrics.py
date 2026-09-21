"""Analytics helpers: VWAP, order-book imbalance, and trade-flow imbalance.

The functions accept either :mod:`exchange_engine.models` dataclasses
(:class:`~exchange_engine.models.Trade`, :class:`~exchange_engine.models.BookLevel`)
or plain mappings/tuples, so they can be reused against the live feed mirror,
the simulation book, or raw payloads. All math is done in
:class:`decimal.Decimal`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Sequence, Tuple, Union

from .models import BookLevel, Trade

TradeLike = Union[Trade, dict]
LevelLike = Union[BookLevel, Tuple[Decimal, Decimal], Sequence]


def _trade_price_size(trade: TradeLike) -> Tuple[Decimal, Decimal]:
    """Return ``(price, size)`` as Decimals from a trade dataclass or dict."""
    if isinstance(trade, Trade):
        return trade.price, trade.size
    return Decimal(str(trade["price"])), Decimal(str(trade["size"]))


def _trade_side(trade: TradeLike) -> str:
    """Return the upper-cased maker side (``"BUY"``/``"SELL"``) of a trade."""
    if isinstance(trade, Trade):
        return trade.side.upper()
    return str(trade.get("side", "")).upper()


def _level_price_qty(level: LevelLike) -> Tuple[Decimal, Decimal]:
    """Return ``(price, quantity)`` as Decimals from a level-like object."""
    if isinstance(level, BookLevel):
        return level.price, level.quantity
    price, quantity = level[0], level[1]
    return Decimal(str(price)), Decimal(str(quantity))


def rolling_vwap(trades: Sequence[TradeLike], window: int = 100) -> Decimal:
    """Volume-weighted average price over the last ``window`` trades.

    Args:
        trades: Ordered sequence of trades (oldest first); the last ``window``
            entries are used.
        window: Number of most-recent trades to include.

    Returns:
        ``sum(price * size) / sum(size)``, or ``0`` when there is no volume.
    """
    recent = list(trades)[-window:]
    numerator = Decimal(0)
    volume = Decimal(0)
    for trade in recent:
        price, size = _trade_price_size(trade)
        numerator += price * size
        volume += size
    if volume == 0:
        return Decimal(0)
    return numerator / volume


def book_imbalance(
    bids: Iterable[LevelLike], asks: Iterable[LevelLike], levels: int = 10
) -> Decimal:
    """Order-book imbalance over the top ``levels`` of each side.

    Args:
        bids: Bid levels ordered best-first.
        asks: Ask levels ordered best-first.
        levels: Number of levels to include from each side.

    Returns:
        ``(bid_qty - ask_qty) / (bid_qty + ask_qty)`` in ``[-1, 1]``, or ``0``
        when both sides are empty.
    """
    bid_qty = sum((_level_price_qty(lvl)[1] for lvl in list(bids)[:levels]), Decimal(0))
    ask_qty = sum((_level_price_qty(lvl)[1] for lvl in list(asks)[:levels]), Decimal(0))
    total = bid_qty + ask_qty
    if total == 0:
        return Decimal(0)
    return (bid_qty - ask_qty) / total


def trade_flow_imbalance(trades: Sequence[TradeLike], window: int = 50) -> Decimal:
    """Signed trade-flow imbalance over the last ``window`` trades.

    Buy volume is attributed to trades whose maker side is ``SELL`` (an
    aggressive buyer lifted the offer), and sell volume to ``BUY`` makers.

    Args:
        trades: Ordered sequence of trades (oldest first).
        window: Number of most-recent trades to include.

    Returns:
        ``(buy_volume - sell_volume) / total_volume`` in ``[-1, 1]``, or ``0``
        when there is no volume.
    """
    recent = list(trades)[-window:]
    buy_volume = Decimal(0)
    sell_volume = Decimal(0)
    for trade in recent:
        _price, size = _trade_price_size(trade)
        if _trade_side(trade) == "SELL":
            buy_volume += size
        else:
            sell_volume += size
    total = buy_volume + sell_volume
    if total == 0:
        return Decimal(0)
    return (buy_volume - sell_volume) / total
