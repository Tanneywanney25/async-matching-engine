"""Property-based tests (Hypothesis) for the matching engine's invariants.

A rule-based state machine drives a book through random limit orders, market orders,
cancels (of live, dead and unknown ids) and re-used order ids, and after EVERY step checks:

* no crossed book: ``best_bid < best_ask`` whenever both sides are populated;
* quantity conservation per order: taker fills + maker fills + remaining quantity (resting
  or cancelled) == submitted quantity, with the status consistent with those numbers;
* price priority: an aggressive order's fills walk the opposite side best price first, each
  level exhausted before the next, never beyond the taker's limit, and the aggregated depth
  always equals the sum of the tracked resting orders;
* time priority within a level: only the oldest resting order at a price can have been
  consumed as a maker (a dedicated test checks the earliest order is filled first);
* idempotent cancel: the second cancel returns False and the observable state is unchanged.

The same machine runs against the naive ReferenceBook (the fuzzer's oracle), which must
satisfy the same invariants. Budget for the whole file: under 20 s.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Dict, List, Optional, Tuple

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from exchange_engine.models import Order, OrderStatus, Side
from exchange_engine.orderbook import OrderBook
from exchange_engine.reference_engine import ReferenceBook

TICK = Decimal("0.01")
RESTING = (OrderStatus.OPEN, OrderStatus.PARTIAL)
UNKNOWN_ID = 1_000_000

# No example database: nothing is written outside the deliverables (``.gitignore`` is not
# ours to edit) and ``print_blob`` still gives a pasteable reproduction on failure.
COMMON = dict(deadline=None, database=None, print_blob=True,
              suppress_health_check=[HealthCheck.too_slow])
GIVEN_SETTINGS = settings(max_examples=150, **COMMON)
MACHINE_SETTINGS = settings(max_examples=100, stateful_step_count=50, **COMMON)

sides = st.sampled_from([Side.BID, Side.ASK])
# A narrow band of 1-cent ticks around $100 so random orders cross often.
prices = st.integers(min_value=9_990, max_value=10_010).map(lambda t: Decimal(t) * TICK)
quantities = st.one_of(
    st.sampled_from([Decimal(q) for q in ("0.25", "0.5", "0.75", "1", "1.5", "2", "3")]),
    st.decimals(min_value=Decimal("0.001"), max_value=Decimal("5"), places=3),
)


@dataclass
class Record:
    """What the test knows about one accepted order, independently of the book."""

    order: Order
    submitted: Decimal
    taker_filled: Decimal
    maker_filled: Decimal
    seq: int  # arrival order; unique even when an order id is re-used


def observable(book, orders) -> tuple:
    """Everything a client can see: top of book, full depth, resting count, statuses."""
    depth = book.get_depth(1_000_000)
    return (
        book.best_bid,
        book.best_ask,
        len(book),
        tuple((lvl.price, lvl.quantity) for lvl in depth["bids"]),
        tuple((lvl.price, lvl.quantity) for lvl in depth["asks"]),
        tuple((o.status, o.quantity) for o in orders),
    )


def expected_level_fills(taker_side: Side, qty: Decimal, limit: Optional[Decimal],
                         levels: List[Tuple[Decimal, Decimal]]) -> List[Tuple[Decimal, Decimal]]:
    """What price priority dictates: walk the aggregated opposite side best price first."""
    out: List[Tuple[Decimal, Decimal]] = []
    remaining = qty
    for price, level_qty in levels:  # already best first
        if remaining <= 0:
            break
        if limit is not None and ((taker_side is Side.BID and price > limit)
                                  or (taker_side is Side.ASK and price < limit)):
            break
        take = min(remaining, level_qty)
        out.append((price, take))
        remaining -= take
    return out


class BookMachine(RuleBasedStateMachine):
    book_factory: ClassVar[type] = OrderBook

    def __init__(self) -> None:
        super().__init__()
        self.book = self.book_factory("PROP")
        self.records: List[Record] = []  # every accepted order, arrival order
        self.latest: Dict[int, Record] = {}  # id -> the newest record under that id
        self.next_id = 1

    # -- helpers -------------------------------------------------------------

    def _resting(self) -> List[Record]:
        return [r for r in self.records if r.order.status in RESTING]

    def _orders(self) -> List[Order]:
        return [r.order for r in self.records]

    def _opposing_levels(self, taker_side: Side) -> List[Tuple[Decimal, Decimal]]:
        depth = self.book.get_depth(1_000_000)
        side = depth["asks"] if taker_side is Side.BID else depth["bids"]
        return [(lvl.price, lvl.quantity) for lvl in side]

    def _check_fills(self, fills, taker_side: Side, taker_id: int, qty: Decimal,
                     limit: Optional[Decimal], levels) -> Decimal:
        assert all(f.order_id == taker_id and f.side is taker_side for f in fills)
        fill_prices = [f.price for f in fills]
        assert fill_prices == sorted(fill_prices, reverse=(taker_side is Side.ASK))
        per_level: List[Tuple[Decimal, Decimal]] = []
        for f in fills:
            if per_level and per_level[-1][0] == f.price:
                per_level[-1] = (f.price, per_level[-1][1] + f.quantity)
            else:
                per_level.append((f.price, f.quantity))
        assert per_level == expected_level_fills(taker_side, qty, limit, levels)
        return sum((f.quantity for f in fills), Decimal(0))

    def _settle_makers(self, before: Dict[int, Decimal], total: Decimal) -> None:
        """Attribute the aggressor's fills to makers through their quantity deltas."""
        consumed = Decimal(0)
        for rec in self.records:
            if rec.seq not in before:
                continue
            delta = before[rec.seq] - rec.order.quantity
            assert delta >= 0
            rec.maker_filled += delta
            consumed += delta
            if rec.order.quantity == 0:
                assert rec.order.status is OrderStatus.FILLED
            elif delta > 0:
                assert rec.order.status is OrderStatus.PARTIAL
        assert consumed == total  # every fill came out of exactly one resting order

    def _submit_limit(self, oid: int, side: Side, price: Decimal, qty: Decimal) -> None:
        levels = self._opposing_levels(side)
        before = {r.seq: r.order.quantity for r in self._resting()}
        order = Order(side, price, qty, id=oid)
        fills = self.book.add_order(order)
        taker = self._check_fills(fills, side, oid, qty, price, levels)
        assert order.quantity == qty - taker  # add_order's own conservation
        if taker == qty:
            assert order.status is OrderStatus.FILLED
        elif taker > 0:
            assert order.status is OrderStatus.PARTIAL
        else:
            assert order.status is OrderStatus.OPEN
        rec = Record(order, qty, taker, Decimal(0), len(self.records))
        self.records.append(rec)
        self.latest[oid] = rec
        self._settle_makers(before, taker)

    # -- rules ---------------------------------------------------------------

    @rule(side=sides, price=prices, qty=quantities)
    def limit_order(self, side: Side, price: Decimal, qty: Decimal) -> None:
        oid, self.next_id = self.next_id, self.next_id + 1
        self._submit_limit(oid, side, price, qty)

    @rule(side=sides, qty=quantities)
    def market_order(self, side: Side, qty: Decimal) -> None:
        levels = self._opposing_levels(side)
        before = {r.seq: r.order.quantity for r in self._resting()}
        oid, self.next_id = self.next_id, self.next_id + 1
        fills = self.book.match_market_order(side, qty, oid)
        taker = self._check_fills(fills, side, oid, qty, None, levels)
        assert taker <= qty
        if taker < qty:  # a remainder is dropped only because the opposite side ran dry
            assert (self.book.best_ask if side is Side.BID else self.book.best_bid) is None
        self._settle_makers(before, taker)

    @rule(data=st.data())
    def cancel(self, data) -> None:
        if self.records and data.draw(st.booleans(), label="cancel a known id"):
            oid = data.draw(st.sampled_from([r.order.id for r in self.records]), label="id")
        else:
            oid = data.draw(st.integers(min_value=UNKNOWN_ID, max_value=UNKNOWN_ID + 20), label="id")
        live = self.latest.get(oid)
        was_resting = live is not None and live.order.status in RESTING
        before = observable(self.book, self._orders())
        assert self.book.cancel_order(oid) is was_resting
        if was_resting:
            assert live.order.status is OrderStatus.CANCELLED
            assert live.order.quantity > 0  # the cancelled remainder is kept, never zeroed
        else:
            assert observable(self.book, self._orders()) == before  # dead or unknown: no change
        # idempotence: a second cancel returns False and changes nothing
        after_first = observable(self.book, self._orders())
        assert self.book.cancel_order(oid) is False
        assert observable(self.book, self._orders()) == after_first

    @precondition(lambda self: bool(self.records))
    @rule(side=sides, price=prices, qty=quantities, data=st.data())
    def reuse_id(self, side: Side, price: Decimal, qty: Decimal, data) -> None:
        """A live id is rejected without side effects; a dead id is accepted again."""
        oid = data.draw(st.sampled_from([r.order.id for r in self.records]), label="id")
        if self.latest[oid].order.status in RESTING:
            before = observable(self.book, self._orders())
            with pytest.raises(ValueError, match="duplicate"):
                self.book.add_order(Order(side, price, qty, id=oid))
            assert observable(self.book, self._orders()) == before
        else:
            self._submit_limit(oid, side, price, qty)

    # -- invariants ----------------------------------------------------------

    @invariant()
    def book_is_never_crossed(self) -> None:
        bid, ask = self.book.best_bid, self.book.best_ask
        if bid is not None and ask is not None:
            assert bid < ask

    @invariant()
    def quantity_is_conserved_per_order(self) -> None:
        for rec in self.records:
            o = rec.order
            assert rec.taker_filled + rec.maker_filled + o.quantity == rec.submitted
            if o.status is OrderStatus.FILLED:
                assert o.quantity == 0
            elif o.status is OrderStatus.OPEN:
                assert rec.taker_filled + rec.maker_filled == 0 and o.quantity == rec.submitted
            elif o.status is OrderStatus.PARTIAL:
                assert 0 < o.quantity < rec.submitted
            else:
                assert o.status is OrderStatus.CANCELLED and o.quantity > 0

    @invariant()
    def depth_equals_the_tracked_resting_orders(self) -> None:
        expected: Dict[Tuple[Side, Decimal], Decimal] = defaultdict(Decimal)
        for rec in self._resting():
            expected[(rec.order.side, rec.order.price)] += rec.order.quantity
        depth = self.book.get_depth(1_000_000)
        bids = [(lvl.price, lvl.quantity) for lvl in depth["bids"]]
        asks = [(lvl.price, lvl.quantity) for lvl in depth["asks"]]
        assert bids == sorted(((p, q) for (s, p), q in expected.items() if s is Side.BID), reverse=True)
        assert asks == sorted((p, q) for (s, p), q in expected.items() if s is Side.ASK)
        assert len(self.book) == len(self._resting())
        assert self.book.best_bid == (bids[0][0] if bids else None)
        assert self.book.best_ask == (asks[0][0] if asks else None)

    @invariant()
    def only_the_oldest_order_at_a_level_has_maker_fills(self) -> None:
        by_level: Dict[Tuple[Side, Decimal], List[Record]] = defaultdict(list)
        for rec in self._resting():
            by_level[(rec.order.side, rec.order.price)].append(rec)
        for group in by_level.values():
            group.sort(key=lambda r: r.seq)
            for rec in group[1:]:
                assert rec.maker_filled == 0, "a younger order was consumed before the oldest"


class OrderBookMachine(BookMachine):
    book_factory = OrderBook


class ReferenceBookMachine(BookMachine):
    book_factory = ReferenceBook


TestOrderBookInvariants = OrderBookMachine.TestCase
TestOrderBookInvariants.settings = MACHINE_SETTINGS
TestReferenceBookInvariants = ReferenceBookMachine.TestCase
TestReferenceBookInvariants.settings = MACHINE_SETTINGS


# --------------------------------------------------------------------------- #
# Direct statements of the four properties
# --------------------------------------------------------------------------- #


@given(side=sides, price=prices,
       resting=st.lists(quantities, min_size=1, max_size=6), aggressor=quantities)
@GIVEN_SETTINGS
def test_earliest_order_at_the_best_level_is_consumed_first(side, price, resting, aggressor) -> None:
    book = OrderBook("PROP")
    makers = [Order(side, price, q, id=i + 1) for i, q in enumerate(resting)]
    for maker in makers:
        assert book.add_order(maker) == []
    behind_price = price - TICK if side is Side.BID else price + TICK
    behind = Order(side, behind_price, Decimal(1), id=99)
    assert book.add_order(behind) == []
    # A limit at exactly the best price cannot reach the level behind it.
    fills = book.add_order(Order(side.opposite, price, aggressor, id=100))
    assert all(f.price == price for f in fills)
    total = sum((f.quantity for f in fills), Decimal(0))
    assert total == min(aggressor, sum(resting, Decimal(0)))
    remaining = total
    for maker, q in zip(makers, resting):  # a filled prefix, at most one partial, the rest untouched
        if remaining >= q:
            assert maker.status is OrderStatus.FILLED and maker.quantity == 0
            remaining -= q
        elif remaining > 0:
            assert maker.status is OrderStatus.PARTIAL and maker.quantity == q - remaining
            remaining = Decimal(0)
        else:
            assert maker.status is OrderStatus.OPEN and maker.quantity == q
    assert behind.status is OrderStatus.OPEN and behind.quantity == 1


@given(orders=st.lists(st.tuples(sides, prices, quantities), min_size=1, max_size=8), data=st.data())
@GIVEN_SETTINGS
def test_cancel_is_idempotent(orders, data) -> None:
    book = OrderBook("PROP")
    placed = [Order(s, p, q, id=i + 1) for i, (s, p, q) in enumerate(orders)]
    for order in placed:
        book.add_order(order)
    target = data.draw(st.sampled_from(placed), label="order to cancel")
    was_resting = target.status in RESTING
    assert book.cancel_order(target.id) is was_resting
    state = observable(book, placed)
    assert book.cancel_order(target.id) is False
    assert observable(book, placed) == state
    assert book.cancel_order(target.id) is False
    assert observable(book, placed) == state


operations = st.one_of(
    st.tuples(st.just("limit"), sides, prices, quantities),
    st.tuples(st.just("market"), sides, quantities),
    st.tuples(st.just("cancel"), st.integers(min_value=1, max_value=40)),
)


@given(ops=st.lists(operations, max_size=40))
@GIVEN_SETTINGS
def test_book_is_never_crossed_and_quantity_is_conserved_globally(ops) -> None:
    book = OrderBook("PROP")
    submitted = taker_filled = all_fills = Decimal(0)
    orders: List[Order] = []
    for i, op in enumerate(ops):
        if op[0] == "limit":
            order = Order(op[1], op[2], op[3], id=i + 1)
            fills = book.add_order(order)
            orders.append(order)
            submitted += op[3]
            taker_filled += sum((f.quantity for f in fills), Decimal(0))
        elif op[0] == "market":
            fills = book.match_market_order(op[1], op[2], i + 1)
        else:
            book.cancel_order(op[1])
            fills = []
        all_fills += sum((f.quantity for f in fills), Decimal(0))
        bid, ask = book.best_bid, book.best_ask
        assert bid is None or ask is None or bid < ask
    # Every fill moves quantity out of exactly one resting limit order (the maker), so:
    # submitted == taker fills of limit orders + all maker fills + what is resting or cancelled.
    remaining = sum((o.quantity for o in orders), Decimal(0))
    assert submitted == taker_filled + all_fills + remaining
