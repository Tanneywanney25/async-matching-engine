"""Differential tests: the real OrderBook against the naive ReferenceBook.

``scripts/fuzz_engine.py`` is imported directly (``tests/conftest.py`` puts ``scripts/`` on
``sys.path``). The main test runs 300 seeded streams; the others check that the fuzzer
itself bites: two deliberately broken books must be caught and the reproduction must be
minimised to the handful of operations that expose the bug.
"""

from __future__ import annotations

import ast
import time
from decimal import Decimal
from typing import List

import pytest

import fuzz_engine
from exchange_engine.models import Fill, Order, Side
from exchange_engine.orderbook import OrderBook
from exchange_engine.reference_engine import ReferenceBook

STREAMS = 300
BUDGET_S = 10.0


def test_300_seeded_streams_never_diverge() -> None:
    t0 = time.perf_counter()
    result = fuzz_engine.run_streams(STREAMS, base_seed=0)
    elapsed = time.perf_counter() - t0
    assert result.streams == STREAMS
    assert result.operations >= STREAMS * fuzz_engine.MIN_OPS
    assert result.ok, "\n\n".join(d.report() for d in result.divergences)
    assert elapsed < BUDGET_S, f"{STREAMS} streams took {elapsed:.1f} s"


def test_streams_are_deterministic_bounded_and_mixed() -> None:
    assert fuzz_engine.generate_stream(7) == fuzz_engine.generate_stream(7)
    assert fuzz_engine.generate_stream(7) != fuzz_engine.generate_stream(8)
    kinds = set()
    reused_ids = unknown_cancels = 0
    for seed in range(50):
        ops = fuzz_engine.generate_stream(seed)
        assert fuzz_engine.MIN_OPS <= len(ops) <= fuzz_engine.MAX_OPS
        seen: set = set()
        for op in ops:
            fuzz_engine.validate_op(op)
            kinds.add(op[0])
            if op[0] == "limit":
                reused_ids += op[1] in seen
                seen.add(op[1])
            elif op[0] == "cancel" and op[1] >= fuzz_engine.UNKNOWN_ID_BASE:
                unknown_cancels += 1
    assert kinds == {"limit", "market", "cancel"}
    assert reused_ids > 0 and unknown_cancels > 0  # duplicate ids and dead-id cancels are exercised


def test_operations_round_trip_as_python_literals(tmp_path) -> None:
    ops = fuzz_engine.generate_stream(3)
    assert ast.literal_eval(repr(ops)) == ops
    path = tmp_path / "ops.txt"
    path.write_text("ops = [\n" + "".join(f"    {op!r},\n" for op in ops) + "]\n", encoding="utf-8")
    assert fuzz_engine.load_ops(path) == ops
    assert fuzz_engine.main(["--replay", str(path)]) == 0


class NewestFirstBook(ReferenceBook):
    """Broken time priority: at a price tie the NEWEST resting order is matched first."""

    def _best(self, side: Side):
        best = None
        for order in self._resting:
            if order.side is not side:
                continue
            if best is None:
                best = order
            elif side is Side.BID and order.price >= best.price:
                best = order
            elif side is Side.ASK and order.price <= best.price:
                best = order
        return best


class TakerPriceBook(OrderBook):
    """Broken fill price: aggressive limit orders are filled at their own price, not the maker's."""

    def add_order(self, order: Order) -> List[Fill]:
        fills = super().add_order(order)
        return [Fill(f.order_id, order.price, f.quantity, f.timestamp, f.side) for f in fills]


@pytest.mark.parametrize("broken, fields, max_ops", [
    # two makers at one price plus one aggressor: the wrong maker order shows in the fill
    # sequence (outcome) or, when a maker is only partly consumed, in the statuses
    (NewestFirstBook, {"outcome", "statuses"}, 3),
    (TakerPriceBook, {"outcome"}, 2),  # one maker plus one aggressor at a different price
])
def test_fuzzer_catches_a_broken_book_and_minimises_the_stream(broken, fields, max_ops) -> None:
    result = fuzz_engine.run_streams(100, base_seed=0, book_factory=broken, max_divergences=1)
    assert len(result.divergences) == 1, "the broken book went unnoticed over 100 streams"
    divergence = result.divergences[0]
    assert divergence.seed is not None and divergence.found["diffs"]
    # the minimised stream still diverges, is 1-minimal and as small as the bug allows
    assert fuzz_engine.find_divergence(divergence.minimized, book_factory=broken) is not None
    assert len(divergence.minimized) <= max_ops
    for i in range(len(divergence.minimized)):
        smaller = divergence.minimized[:i] + divergence.minimized[i + 1:]
        assert not smaller or fuzz_engine.find_divergence(smaller, book_factory=broken) is None
    found_fields = {name for name, _real, _ref in divergence.minimized_found["diffs"]}
    assert found_fields & fields, found_fields
    report = divergence.report()
    assert f"seed={divergence.seed}" in report and "ops = [" in report
    assert any(field in report for field in found_fields & fields)


def test_replay_of_a_divergent_literal_fails_with_a_report(tmp_path, capsys) -> None:
    # A hand-written stream that a taker-priced book gets wrong, replayed through the CLI
    # against a book that is correct must pass; the broken book must fail with the report.
    ops = [("limit", 1, "ask", "64000.00", "1"), ("limit", 2, "bid", "64000.05", "1")]
    path = tmp_path / "ops.txt"
    path.write_text(repr(ops), encoding="utf-8")
    assert fuzz_engine.main(["--replay", str(path)]) == 0
    found = fuzz_engine.find_divergence(ops, book_factory=TakerPriceBook)
    assert found is not None and found["index"] == 1
    outcome = dict((name, (real, ref)) for name, real, ref in found["diffs"])["outcome"]
    assert outcome[0][1][0][2] == Decimal("64000.05") and outcome[1][1][0][2] == Decimal("64000.00")


def test_malformed_operations_are_rejected() -> None:
    with pytest.raises(ValueError):
        fuzz_engine.validate_op(("limit", 1, "bid", "100"))
    with pytest.raises(ValueError):
        fuzz_engine.validate_op(("market", 1, "buy", "1"))
    with pytest.raises(ValueError):
        fuzz_engine.validate_op(("hold", 1))
