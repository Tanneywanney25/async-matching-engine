"""An order stream derived from the recorded archive: book changes and trades of one product.

Replaying recorded snapshots and trades in time order through the Python reference engine:

* each trade becomes a market order on the aggressor's side (the archive stores Coinbase's maker
  side, so a maker BUY means a seller took the bid) for the traded size;
* each snapshot is reconciled with the engine's book. Within the snapshot's price range (bids at
  or above its lowest bid, asks at or below its highest ask) every level whose size shrank or
  vanished loses orders, newest first, through cancels (an overshoot is re-added at the same
  price), and every level whose size grew gains one limit order for the difference. All removals
  come before any addition, so an added order can cross only a stale level outside the range the
  snapshot reconciled; the generator counts any that do (``limit_orders_matched``).

The engine is the generator's state, so every cancel names an order that is resting at that
moment and the stream replays identically through any book with the same semantics. Sizes become
integer lots of 1e-8 once, here. ``derive`` returns plain tuples:
``("limit", id, "bid"|"ask", ticks, lots)``, ``("market", id, side, lots)``, ``("cancel", id)``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

TICK = Decimal("0.01")
LOT = Decimal("0.00000001")
LOTS_PER_UNIT = 100_000_000

Op = Tuple


def _lots(size: float) -> int:
    return int(round(float(size) * LOTS_PER_UNIT))


def _events(arch: dict, product_index: int) -> Iterator[Tuple[int, str, int]]:
    """``(timestamp, "snap"|"trade", row)`` for one product, in time order (snapshots first on ties)."""
    s_rows = np.flatnonzero(arch["snap_product"] == product_index)
    t_rows = np.flatnonzero(arch["trade_product"] == product_index)
    s_ts, t_ts = arch["snap_ts"][s_rows], arch["trade_ts"][t_rows]
    s_order, t_order = np.argsort(s_ts, kind="stable"), np.argsort(t_ts, kind="stable")
    i = j = 0
    while i < len(s_order) or j < len(t_order):
        if j >= len(t_order) or (i < len(s_order) and s_ts[s_order[i]] <= t_ts[t_order[j]]):
            yield int(s_ts[s_order[i]]), "snap", int(s_rows[s_order[i]])
            i += 1
        else:
            yield int(t_ts[t_order[j]]), "trade", int(t_rows[t_order[j]])
            j += 1


class _Generator:
    """The reference engine plus the bookkeeping that turns snapshots and trades into operations."""

    def __init__(self) -> None:
        from exchange_engine.models import Order, Side
        from exchange_engine.orderbook import OrderBook

        self.Order, self.Side = Order, Side
        self.book = OrderBook("REALISTIC")
        self.ops: List[Op] = []
        self.next_id = 1
        self.counts = {"limit": 0, "market": 0, "cancel": 0}
        self.limits_matched = self.limit_fills = 0  # limit orders that crossed on arrival, and their fills

    def _emit(self, op: Op) -> None:
        self.ops.append(op)
        self.counts[op[0]] += 1

    def limit(self, side: str, ticks: int, lots: int) -> None:
        oid, self.next_id = self.next_id, self.next_id + 1
        s = self.Side.BID if side == "bid" else self.Side.ASK
        fills = self.book.add_order(self.Order(s, ticks * TICK, lots * LOT, id=oid))
        if fills:
            self.limits_matched += 1
            self.limit_fills += len(fills)
        self._emit(("limit", oid, side, ticks, lots))

    def market(self, side: str, lots: int) -> None:
        oid, self.next_id = self.next_id, self.next_id + 1
        self.book.match_market_order(self.Side.BID if side == "bid" else self.Side.ASK, lots * LOT, oid)
        self._emit(("market", oid, side, lots))

    def cancel(self, oid: int) -> None:
        self.book.cancel_order(oid)
        self._emit(("cancel", oid))

    def levels_in_range(self, side: str, lo: int, hi: int) -> Dict[int, int]:
        """``{ticks: lots}`` of the engine's levels from the best price out to the snapshot's last level.

        Read through the sorted tree, so stale deep levels outside the snapshot cost nothing.
        """
        if side == "bid":  # keys are negated prices, best first; stop below the snapshot's lowest bid
            keys = self.book._bids.irange(maximum=-(lo * TICK))  # noqa: SLF001 - the generator reads its own state
            qty, sign = self.book._bid_qty, -1  # noqa: SLF001
        else:
            keys = self.book._asks.irange(maximum=hi * TICK)  # noqa: SLF001
            qty, sign = self.book._ask_qty, 1  # noqa: SLF001
        out = {}
        for key in keys:
            price = sign * key
            out[int(price / TICK)] = int(qty[price] / LOT)
        return out

    def orders_at(self, side: str, ticks: int) -> list:
        tree = self.book._bids if side == "bid" else self.book._asks  # noqa: SLF001
        key = -(ticks * TICK) if side == "bid" else ticks * TICK
        return list(tree.get(key, ()))

    def reconcile(self, target: Dict[str, Dict[int, int]]) -> None:
        adds: List[Tuple[str, int, int]] = []
        for side in ("bid", "ask"):
            want = target[side]
            if not want:
                continue
            lo, hi = min(want), max(want)
            cur = self.levels_in_range(side, lo, hi)
            for price in sorted(set(cur) | set(want)):
                have, need = cur.get(price, 0), want.get(price, 0)
                if need < have:
                    removed = 0
                    for order in reversed(self.orders_at(side, price)):  # newest first
                        if removed >= have - need:
                            break
                        removed += int(order.quantity / LOT)
                        self.cancel(order.id)
                    if removed > have - need:
                        adds.append((side, price, removed - (have - need)))
                elif need > have:
                    adds.append((side, price, need - have))
        for side, price, lots in adds:
            self.limit(side, price, lots)


def derive(arch: dict, product: str = "BTC-USD", max_ops: Optional[int] = None) -> dict:
    """Operations for ``product`` from the archive, plus what went into them."""
    products = list(arch["meta"]["products"])
    pi = products.index(product)
    gen = _Generator()
    n_snaps = n_trades = 0
    first_ts = last_ts = None
    levels = arch["bid_px"].shape[0]
    for ts, kind, row in _events(arch, pi):
        first_ts = ts if first_ts is None else first_ts
        last_ts = ts
        if kind == "trade":
            lots = _lots(arch["trade_sz"][row])
            if lots > 0:
                gen.market("ask" if int(arch["trade_side"][row]) == 1 else "bid", lots)
            n_trades += 1
        else:
            target = {"bid": {}, "ask": {}}
            for side, px, sz in (("bid", arch["bid_px"], arch["bid_sz"]), ("ask", arch["ask_px"], arch["ask_sz"])):
                for lvl in range(levels):
                    p, q = int(px[lvl, row]), _lots(sz[lvl, row])
                    if p > 0 and q > 0:
                        target[side][p] = q
            gen.reconcile(target)
            n_snaps += 1
        if max_ops is not None and len(gen.ops) >= max_ops:
            break
    span_s = (last_ts - first_ts) / 1e9 if first_ts is not None else 0.0  # timestamps are nanoseconds
    return {"ops": gen.ops, "counts": dict(gen.counts), "snapshots_used": n_snaps, "trades_used": n_trades,
            "product": product, "resting_at_end": len(gen.book), "span_seconds": span_s,
            "limit_orders_matched": gen.limits_matched, "limit_fills": gen.limit_fills,
            "description": (f"derived from the first {span_s / 60:,.1f} minutes of the recorded {product} book: "
                            f"{n_snaps:,} snapshots reconciled into limit orders and cancels, {n_trades:,} trades "
                            f"replayed as market orders on the aggressor's side")}

