"""Differential fuzzer: the real OrderBook against the naive list-based ReferenceBook.

Runs N seeded random order streams (default 10,000; each 30-80 operations mixing limit
orders near a mid price, market orders, cancels of live, dead and unknown ids, and
re-used order ids) through :class:`exchange_engine.orderbook.OrderBook` and
:class:`exchange_engine.reference_engine.ReferenceBook` and compares, after EVERY
operation: the operation's outcome (the fills as ``(taker id, side, price, quantity)``
in order, the cancel result, or the exception type), best bid and ask, depth(5), the
number of resting orders, and the status and remaining quantity of every order accepted
so far in the stream.

On a divergence the stream is minimised by delta debugging (ddmin: drop chunks of
operations while a divergence persists, then a one-at-a-time pass so the result is
1-minimal) and the run fails with a reproduction: the seed, the minimised operation
list as Python literals, and the differing values.

Run with::

    python scripts/fuzz_engine.py --streams 10000          # the full run, exit 1 on divergence
    python scripts/fuzz_engine.py --seed 1234              # one stream
    python scripts/fuzz_engine.py --replay ops.txt         # a file holding a literal list of operations

Operations are plain tuples so a reproduction can be pasted straight back in::

    ("limit", order_id, "bid" | "ask", "<price>", "<quantity>")
    ("market", order_id, "bid" | "ask", "<quantity>")
    ("cancel", order_id)

Prices and quantities are strings (exact Decimal literals). Streams are a pure function
of their seed, so a seed alone reproduces a run on any machine.
"""

from __future__ import annotations

import argparse
import ast
import random
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Allow running directly from a checkout without installing the package.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exchange_engine.models import Fill, Order, Side  # noqa: E402
from exchange_engine.orderbook import OrderBook  # noqa: E402
from exchange_engine.reference_engine import ReferenceBook  # noqa: E402

Op = Tuple[object, ...]
BookFactory = Callable[[str], object]
Diff = Tuple[str, object, object]  # (field, real value, reference value)

DEFAULT_STREAMS = 10_000
MIN_OPS, MAX_OPS = 30, 80
DEPTH_LEVELS = 5
TICK = Decimal("0.01")
MID_TICKS = 6_400_000  # $64,000.00
NEAR_BAND_TICKS = 12  # most orders land within 12 ticks of the mid so the two sides cross often
FAR_BAND_TICKS = (50, 500)  # a few land far away and build levels that only a sweep reaches
QUANTITIES = ("0.25", "0.5", "0.75", "1", "1.5", "2", "3")  # exact fractions: partial fills are frequent
BIG_QUANTITIES = ("4", "6", "10")  # market sweeps across several levels, sometimes the whole side
UNKNOWN_ID_BASE = 1_000_000  # cancels of ids that were never submitted
SIDES: Dict[str, Side] = {"bid": Side.BID, "ask": Side.ASK}


# --------------------------------------------------------------------------- #
# Stream generation (a pure function of the seed)
# --------------------------------------------------------------------------- #


def _price(rng: random.Random) -> str:
    if rng.random() < 0.05:
        lo, hi = FAR_BAND_TICKS
        offset = rng.randint(lo, hi) * rng.choice((-1, 1))
    else:
        offset = rng.randint(-NEAR_BAND_TICKS, NEAR_BAND_TICKS)
    return str(Decimal(MID_TICKS + offset) * TICK)


def _quantity(rng: random.Random, market: bool = False) -> str:
    r = rng.random()
    if market and r < 0.25:
        return rng.choice(BIG_QUANTITIES)
    if r < 0.85:
        return rng.choice(QUANTITIES)
    return f"{rng.uniform(0.001, 2.5):.3f}"


def _side(rng: random.Random) -> str:
    return "bid" if rng.random() < 0.5 else "ask"


def generate_stream(seed: int, min_ops: int = MIN_OPS, max_ops: int = MAX_OPS) -> List[Op]:
    """Return the operation list for ``seed``: 55% fresh limit orders, 7% limit orders that
    re-use an earlier id (rejected while that id rests, accepted once it is dead), 18% market
    orders, 14% cancels of an id used so far (live or dead) and 6% cancels of unknown ids."""
    rng = random.Random(seed)
    n_ops = rng.randint(min_ops, max_ops)
    ops: List[Op] = []
    used_ids: List[int] = []
    next_id = 1
    for _ in range(n_ops):
        r = rng.random()
        if r < 0.55 or not used_ids:
            oid, next_id = next_id, next_id + 1
            used_ids.append(oid)
            ops.append(("limit", oid, _side(rng), _price(rng), _quantity(rng)))
        elif r < 0.62:
            ops.append(("limit", rng.choice(used_ids), _side(rng), _price(rng), _quantity(rng)))
        elif r < 0.80:
            oid, next_id = next_id, next_id + 1
            ops.append(("market", oid, _side(rng), _quantity(rng, market=True)))
        elif r < 0.94:
            ops.append(("cancel", rng.choice(used_ids)))
        else:
            ops.append(("cancel", UNKNOWN_ID_BASE + rng.randint(0, 99)))
    return ops


# --------------------------------------------------------------------------- #
# Applying and observing
# --------------------------------------------------------------------------- #


def validate_op(op: Op) -> None:
    """Raise ``ValueError`` for an operation that is not one of the three literal shapes."""
    shape = {"limit": 5, "market": 4, "cancel": 2}.get(op[0] if op else None)
    if shape is None or len(op) != shape:
        raise ValueError(f"malformed operation {op!r}")
    if op[0] in ("limit", "market") and op[2] not in SIDES:
        raise ValueError(f"side must be 'bid' or 'ask' in {op!r}")


def _fill_key(fill: Fill) -> Tuple[int, str, Decimal, Decimal]:
    return (fill.order_id, fill.side.value, fill.price, fill.quantity)


def apply_op(book, accepted: List[Order], op: Op) -> Tuple[object, ...]:
    """Apply one operation to ``book`` and return its comparable outcome.

    ``accepted`` collects every Order the book accepted (its status and remaining quantity
    are compared after every later operation). An exception IS the outcome: both books must
    raise the same type, so it is returned rather than propagated.
    """
    validate_op(op)
    kind = op[0]
    try:
        if kind == "limit":
            _, oid, side, price, qty = op
            order = Order(SIDES[side], Decimal(price), Decimal(qty), id=int(oid))
            fills = book.add_order(order)
            accepted.append(order)
            return ("fills", tuple(_fill_key(f) for f in fills))
        if kind == "market":
            _, oid, side, qty = op
            fills = book.match_market_order(SIDES[side], Decimal(qty), int(oid))
            return ("fills", tuple(_fill_key(f) for f in fills))
        return ("cancelled", bool(book.cancel_order(int(op[1]))))
    except Exception as exc:  # noqa: BLE001 - the exception type is part of the contract
        return ("error", type(exc).__name__)


def observe(book, accepted: Sequence[Order]) -> Dict[str, object]:
    """Everything compared after an operation besides the operation's own outcome."""
    depth = book.get_depth(DEPTH_LEVELS)
    return {
        "best_bid": book.best_bid,
        "best_ask": book.best_ask,
        "depth_bids": tuple((lvl.price, lvl.quantity) for lvl in depth["bids"]),
        "depth_asks": tuple((lvl.price, lvl.quantity) for lvl in depth["asks"]),
        "resting_orders": len(book),
        "statuses": tuple((o.id, o.status.value, o.quantity) for o in accepted),
    }


def find_divergence(
    ops: Sequence[Op],
    book_factory: BookFactory = OrderBook,
    ref_factory: BookFactory = ReferenceBook,
) -> Optional[dict]:
    """Replay ``ops`` through both books; return ``{"index", "op", "diffs"}`` at the first
    operation after which they disagree, or ``None`` when they agree throughout."""
    real, ref = book_factory("FUZZ"), ref_factory("FUZZ")
    real_accepted: List[Order] = []
    ref_accepted: List[Order] = []
    for index, op in enumerate(ops):
        out_real = apply_op(real, real_accepted, op)
        out_ref = apply_op(ref, ref_accepted, op)
        diffs: List[Diff] = []
        if out_real != out_ref:
            diffs.append(("outcome", out_real, out_ref))
        obs_real, obs_ref = observe(real, real_accepted), observe(ref, ref_accepted)
        for key, value in obs_real.items():
            if value != obs_ref[key]:
                diffs.append((key, value, obs_ref[key]))
        if diffs:
            return {"index": index, "op": op, "diffs": diffs}
    return None


# --------------------------------------------------------------------------- #
# Delta debugging
# --------------------------------------------------------------------------- #


def ddmin(ops: List[Op], failing: Callable[[List[Op]], bool]) -> List[Op]:
    """Zeller's ddmin: drop chunks of ``ops`` while ``failing`` still holds, then a
    one-at-a-time pass so no single operation can be removed from the result."""
    ops = list(ops)
    n = 2
    while len(ops) >= 2:
        size = (len(ops) + n - 1) // n
        chunks = [ops[i:i + size] for i in range(0, len(ops), size)]
        reduced = False
        for part in chunks:  # reduce to a subset
            if failing(part):
                ops, n, reduced = part, 2, True
                break
        if not reduced:  # reduce to a complement
            for i in range(len(chunks)):
                complement = [op for j, part in enumerate(chunks) if j != i for op in part]
                if complement and failing(complement):
                    ops, n, reduced = complement, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(ops):
                break
            n = min(2 * n, len(ops))
    i = 0
    while i < len(ops):
        candidate = ops[:i] + ops[i + 1:]
        if candidate and failing(candidate):
            ops = candidate
        else:
            i += 1
    return ops


def _format_diffs(diffs: Sequence[Diff], indent: str = "  ") -> List[str]:
    return [f"{indent}{field}: real={real!r} reference={ref!r}" for field, real, ref in diffs]


@dataclass
class Divergence:
    """A stream on which the two books disagreed, with its minimised reproduction."""

    seed: Optional[int]
    ops: List[Op]
    found: dict  # {"index", "op", "diffs"} on the original stream
    minimized: List[Op]
    minimized_found: dict  # the same on the minimised stream

    def report(self) -> str:
        where = f"seed={self.seed}" if self.seed is not None else "replayed stream"
        lines = [
            f"DIVERGENCE {where} at operation {self.found['index'] + 1} of {len(self.ops)}: "
            f"{self.found['op']!r}",
            *_format_diffs(self.found["diffs"]),
            f"minimized to {len(self.minimized)} operation(s); replay with "
            f"`python scripts/fuzz_engine.py --replay <file holding the list below>`:",
            "ops = [",
            *[f"    {op!r}," for op in self.minimized],
            "]",
            f"  divergence at operation {self.minimized_found['index'] + 1} of "
            f"{len(self.minimized)}: {self.minimized_found['op']!r}",
            *_format_diffs(self.minimized_found["diffs"], indent="    "),
        ]
        return "\n".join(lines)


def minimize_divergence(
    seed: Optional[int],
    ops: Sequence[Op],
    found: dict,
    book_factory: BookFactory = OrderBook,
    ref_factory: BookFactory = ReferenceBook,
    minimize: bool = True,
) -> Divergence:
    """Build the :class:`Divergence` for ``ops`` (minimising it unless ``minimize`` is false)."""
    ops = list(ops)
    if not minimize:
        return Divergence(seed, ops, found, ops, found)

    def failing(candidate: List[Op]) -> bool:
        return find_divergence(candidate, book_factory, ref_factory) is not None

    minimized = ddmin(ops, failing)
    minimized_found = find_divergence(minimized, book_factory, ref_factory)
    assert minimized_found is not None  # ddmin only ever returns failing candidates
    return Divergence(seed, ops, found, minimized, minimized_found)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


@dataclass
class FuzzResult:
    streams: int
    operations: int
    divergences: List[Divergence]
    seconds: float

    @property
    def ok(self) -> bool:
        return not self.divergences

    def summary(self) -> str:
        return (f"streams={self.streams} operations={self.operations} "
                f"divergences={len(self.divergences)} seconds={self.seconds:.1f}")


def run_streams(
    n_streams: int = DEFAULT_STREAMS,
    base_seed: int = 0,
    *,
    min_ops: int = MIN_OPS,
    max_ops: int = MAX_OPS,
    book_factory: BookFactory = OrderBook,
    ref_factory: BookFactory = ReferenceBook,
    minimize: bool = True,
    max_divergences: Optional[int] = None,
    progress_every: int = 0,
) -> FuzzResult:
    """Run seeds ``base_seed .. base_seed + n_streams - 1`` and collect every divergence
    (stopping early after ``max_divergences`` of them when that is set)."""
    t0 = time.perf_counter()
    operations = 0
    streams = 0
    divergences: List[Divergence] = []
    for k in range(n_streams):
        seed = base_seed + k
        ops = generate_stream(seed, min_ops, max_ops)
        operations += len(ops)
        streams += 1
        found = find_divergence(ops, book_factory, ref_factory)
        if found is not None:
            divergences.append(minimize_divergence(seed, ops, found, book_factory, ref_factory, minimize))
            if max_divergences and len(divergences) >= max_divergences:
                break
        if progress_every and streams % progress_every == 0:
            print(f"  {streams:,} streams, {operations:,} operations, "
                  f"{len(divergences)} divergence(s), {time.perf_counter() - t0:.1f} s", flush=True)
    return FuzzResult(streams, operations, divergences, time.perf_counter() - t0)


def load_ops(path: Path) -> List[Op]:
    """Read a file holding a Python literal list of operations (``ops = [...]`` is accepted)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("ops"):
        text = text.split("=", 1)[1]
    ops = ast.literal_eval(text)
    if not isinstance(ops, list):
        raise ValueError("replay file must hold a list of operation tuples")
    ops = [tuple(op) for op in ops]
    for op in ops:
        validate_op(op)
    return ops


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Differential fuzzer: OrderBook vs ReferenceBook")
    parser.add_argument("--streams", type=int, default=DEFAULT_STREAMS, help="number of seeded streams")
    parser.add_argument("--base-seed", type=int, default=0, help="first seed; stream k uses base + k")
    parser.add_argument("--seed", type=int, default=None, help="run this single seed only")
    parser.add_argument("--replay", type=Path, default=None, help="file holding a literal operation list")
    parser.add_argument("--min-ops", type=int, default=MIN_OPS)
    parser.add_argument("--max-ops", type=int, default=MAX_OPS)
    parser.add_argument("--no-minimize", action="store_true", help="report the raw stream, skip ddmin")
    parser.add_argument("--max-reports", type=int, default=5, help="print at most this many reproductions")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="progress line every N streams (0 = off)")
    args = parser.parse_args(argv)

    if args.replay is not None:
        ops = load_ops(args.replay)
        found = find_divergence(ops)
        if found is None:
            print(f"no divergence over {len(ops)} replayed operation(s)")
            return 0
        print(minimize_divergence(None, ops, found, minimize=not args.no_minimize).report())
        return 1

    if args.seed is not None:
        result = run_streams(1, args.seed, min_ops=args.min_ops, max_ops=args.max_ops,
                             minimize=not args.no_minimize)
    else:
        print(f"fuzzing OrderBook against ReferenceBook: {args.streams:,} streams, "
              f"{args.min_ops}-{args.max_ops} operations each, seeds {args.base_seed}..", flush=True)
        result = run_streams(args.streams, args.base_seed, min_ops=args.min_ops, max_ops=args.max_ops,
                             minimize=not args.no_minimize, progress_every=args.progress_every)

    for divergence in result.divergences[: args.max_reports]:
        print(divergence.report())
        print()
    if len(result.divergences) > args.max_reports:
        print(f"... {len(result.divergences) - args.max_reports} more divergence(s) not shown")
    print(("FAIL " if result.divergences else "OK ") + result.summary())
    return 1 if result.divergences else 0


if __name__ == "__main__":
    sys.exit(main())
