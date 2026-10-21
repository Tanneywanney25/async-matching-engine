"""Latency of the C++20 order book core, natively and through its pybind11 binding.

    python scripts/benchmark_native.py --results-dir <dir> [--orders 100000] [--cancels 10000]
                                       [--build-dir build/native] [--passes 5]

Replays the order stream of ``scripts/benchmark.py`` (same seed, same random calls, so the same
100,000 limit orders around $64,000 and the same 10,000 cancels of resting ids) four ways, all
in this process on this machine:

* ``native``: ``build/native/bench_native``, the C++ harness, with no Python in the loop (it
  reads the stream from a binary file; each operation timed with two steady_clock reads);
* ``binding``: ``exchange_engine._native.Book`` called from Python with integer ticks and lots
  (``perf_counter_ns`` around each call, as ``scripts/benchmark.py`` times the Python engine);
* ``binding_decimal``: :class:`exchange_engine.native_book.NativeOrderBook` with Decimal Orders,
  the interface the differential fuzzer drives;
* ``python_engine``: :class:`exchange_engine.orderbook.OrderBook`, the reference implementation,
  timed the same way here so all four sit side by side.

Orders are built before their timed region, as in ``scripts/benchmark.py``. Fill counts and the
number of resting orders must agree across all four, or the run says so (``consistent``).
Percentiles are nearest-rank, identical to ``scripts/benchmark.py``, with p99.9 added. Writes
``<dir>/metrics.json`` (section ``native_engine``) and the raw latency arrays under ``<dir>/raw``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import struct
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exchange_engine.models import Order, Side  # noqa: E402
from exchange_engine.orderbook import OrderBook  # noqa: E402

TICK = Decimal("0.01")
LOT = Decimal("0.00000001")
WINDOW_TICKS = 1 << 18
RAW_NAME = "native_engine_latency_us.json"


# --------------------------------------------------------------------------- #
# The stream
# --------------------------------------------------------------------------- #


def synthetic_stream(n_orders: int, n_cancels: int, mid: float = 64_000.0, seed: int = 42) -> dict:
    """``scripts/benchmark.py``'s stream: the same random calls in the same order, ids 1..n.

    The cancel list is what ``benchmark.py`` cancels: ids that rested when added, shuffled with
    ``Random(seed + 1)``, the first ``n_cancels`` of them (some have been filled since).
    """
    rng = random.Random(seed)
    sides: List[Side] = []
    prices: List[Decimal] = []
    qtys: List[Decimal] = []
    for _ in range(n_orders):
        sides.append(Side.BID if rng.random() < 0.5 else Side.ASK)
        prices.append(Decimal(f"{rng.gauss(mid, mid * 0.01):.2f}"))
        qtys.append(Decimal(f"{rng.uniform(0.001, 2.0):.6f}"))
    book = OrderBook("STREAM")
    resting: List[int] = []
    for i in range(n_orders):
        order = Order(sides[i], prices[i], qtys[i], id=i + 1)
        book.add_order(order)
        if order.quantity > 0:
            resting.append(order.id)
    random.Random(seed + 1).shuffle(resting)
    return {"kind": "synthetic", "seed": seed, "mid": mid, "ids": list(range(1, n_orders + 1)), "sides": sides,
            "prices": prices, "qtys": qtys, "cancels": resting[:n_cancels],
            "description": f"scripts/benchmark.py's stream: {n_orders:,} limit orders, price gauss(mid, 1% of mid) to the "
                           f"cent, size uniform(0.001, 2.0) to 1e-6, seed {seed}; {n_cancels:,} cancels of rested ids"}


def to_ints(stream: dict) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Ticks and lots, the only place a Decimal becomes an integer."""
    ticks = [int(p / TICK) for p in stream["prices"]]
    lots = [int(q / LOT) for q in stream["qtys"]]
    for p, t in zip(stream["prices"], ticks):
        if t * TICK != p:
            raise ValueError(f"price {p} is not a whole number of ticks")
    sides = [0 if s is Side.BID else 1 for s in stream["sides"]]
    return list(stream["ids"]), sides, ticks, lots


def write_stream(path: Path, stream: dict) -> None:
    ids, sides, ticks, lots = to_ints(stream)
    cancels = stream["cancels"]
    with open(path, "wb") as fh:
        fh.write(b"EESTREAM" + struct.pack("<IIQQ", 1, 0, len(ids), len(cancels)))
        for oid, s, t, q in zip(ids, sides, ticks, lots):
            fh.write(struct.pack("<qqqB7x", oid, t, q, s))
        fh.write(struct.pack(f"<{len(cancels)}q", *cancels))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def percentile(sorted_us: Sequence[float], pct: float) -> float:
    """Nearest rank, identical to ``scripts/benchmark.py``."""
    if not sorted_us:
        return 0.0
    rank = max(0, min(len(sorted_us) - 1, int(round(pct / 100 * len(sorted_us))) - 1))
    return float(sorted_us[rank])


def stats(us: Sequence[float]) -> dict:
    xs = sorted(us)
    return {"n": len(xs), "mean_us": sum(xs) / len(xs) if xs else 0.0, "p50_us": percentile(xs, 50),
            "p95_us": percentile(xs, 95), "p99_us": percentile(xs, 99), "p999_us": percentile(xs, 99.9),
            "max_us": xs[-1] if xs else 0.0}


# --------------------------------------------------------------------------- #
# Build and measure
# --------------------------------------------------------------------------- #


def build_native(build_dir: Path) -> dict:
    """Configure and build ``native/`` (tests, harness, binding); returns versions and timings."""
    t0 = time.perf_counter()
    cfg = ["cmake", "-S", str(REPO / "native"), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
           f"-DPython_EXECUTABLE={sys.executable}"]
    for cmd in (cfg, ["cmake", "--build", str(build_dir), "--parallel", str(os.cpu_count() or 2)]):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    tests = subprocess.run(["ctest", "--test-dir", str(build_dir), "--output-on-failure"], capture_output=True,
                           text=True, timeout=600)
    cxx = subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    return {"seconds": round(time.perf_counter() - t0, 1), "ctest_ok": tests.returncode == 0,
            "ctest_output": tests.stdout[-2000:], "compiler": cxx[0] if cxx else "", "flags": "-O3 (CMake Release), C++20",
            **native_tests(build_dir)}


def native_tests(build_dir: Path) -> dict:
    """Run the two native test binaries and keep the figures they print."""
    import re

    out: dict = {}
    for name in ("test_order_book", "test_no_alloc"):
        exe = build_dir / (name + (".exe" if os.name == "nt" else ""))
        proc = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
        out[name] = {"ok": proc.returncode == 0, "output": (proc.stdout + proc.stderr).strip()[-600:]}
        m = re.search(r"(\d+) checks passed", proc.stdout)
        if m:
            out[name]["checks"] = int(m.group(1))
        m = re.search(r"(\d+) operations, (\d+) fills, .*?: (\d+) allocations, (\d+) deallocations", proc.stdout)
        if m:
            out[name].update({"operations": int(m.group(1)), "fills": int(m.group(2)), "allocations": int(m.group(3)),
                              "deallocations": int(m.group(4))})
    return out


def fuzz(streams: int) -> dict:
    """The differential fuzzer against the core: same streams and comparisons as scripts/fuzz_engine.py."""
    sys.path.insert(0, str(REPO / "scripts"))
    import fuzz_engine
    from exchange_engine.native_book import NativeOrderBook

    result = fuzz_engine.run_streams(streams, book_factory=NativeOrderBook, minimize=False)
    return {"streams": result.streams, "operations": result.operations, "divergences": len(result.divergences),
            "seconds": round(result.seconds, 1), "reference": "exchange_engine.reference_engine.ReferenceBook",
            "first_divergence": result.divergences[0].report()[:2000] if result.divergences else None}


def run_native(build_dir: Path, stream_file: Path, out_file: Path, passes: int) -> dict:
    exe = build_dir / ("bench_native.exe" if os.name == "nt" else "bench_native")
    proc = subprocess.run([str(exe), str(stream_file), str(out_file), "--passes", str(passes)], capture_output=True,
                          text=True, timeout=1200)
    if proc.returncode != 0:
        raise RuntimeError(f"bench_native failed: {proc.stderr[-2000:]}")
    return json.loads(out_file.read_text(encoding="utf-8"))


def time_binding(stream: dict) -> dict:
    from exchange_engine import _native

    ids, sides, ticks, lots = to_ints(stream)
    book = _native.Book(WINDOW_TICKS, max(len(ids), 1024), True)
    add_us, fills = [], 0
    for oid, s, t, q in zip(ids, sides, ticks, lots):
        t0 = time.perf_counter_ns()
        fills += book.add_limit_count(oid, s, t, q)
        add_us.append((time.perf_counter_ns() - t0) / 1000.0)
    resting = len(book)
    cancel_us, found = [], 0
    for oid in stream["cancels"]:
        t0 = time.perf_counter_ns()
        found += book.cancel(oid)
        cancel_us.append((time.perf_counter_ns() - t0) / 1000.0)
    return {"add_us": add_us, "cancel_us": cancel_us, "fills": fills, "resting_after_adds": resting, "cancels_found": found}


def time_orders(stream: dict, book) -> dict:
    """Time ``add_order`` and ``cancel_order`` on any book with the OrderBook interface."""
    orders = [Order(s, p, q, id=oid) for oid, s, p, q in zip(stream["ids"], stream["sides"], stream["prices"], stream["qtys"])]
    add_us, fills = [], 0
    for order in orders:
        t0 = time.perf_counter_ns()
        got = book.add_order(order)
        add_us.append((time.perf_counter_ns() - t0) / 1000.0)
        fills += len(got)
    resting = len(book)
    cancel_us, found = [], 0
    for oid in stream["cancels"]:
        t0 = time.perf_counter_ns()
        found += book.cancel_order(oid)
        cancel_us.append((time.perf_counter_ns() - t0) / 1000.0)
    return {"add_us": add_us, "cancel_us": cancel_us, "fills": fills, "resting_after_adds": resting, "cancels_found": found}


def cpu_model() -> str:
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        import benchmark

        return benchmark._cpu_model()
    except Exception:  # noqa: BLE001
        return platform.processor() or platform.machine()


def measure(stream: dict, results_dir: Path, build_dir: Path, passes: int, label: str = "synthetic") -> dict:
    """All four measurements of one stream; returns the section dict and writes the raw arrays."""
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stream_file = results_dir / f"stream_{label}.bin"
    write_stream(stream_file, stream)
    native = run_native(build_dir, stream_file, raw_dir / f"native_{label}_harness.json", passes)
    binding = time_binding(stream)
    from exchange_engine.native_book import NativeOrderBook

    dec = time_orders(stream, NativeOrderBook("BENCH", window_ticks=WINDOW_TICKS, order_capacity=max(len(stream["ids"]), 1024),
                                              prefault=True))
    py = time_orders(stream, OrderBook("BENCH"))
    counts = {name: (r["fills"], r["resting_after_adds"], r["cancels_found"])
              for name, r in (("native", native), ("binding", binding), ("binding_decimal", dec), ("python_engine", py))}
    raw_rel = f"raw/{label}_{RAW_NAME}"
    (results_dir / raw_rel).write_text(json.dumps({
        "unit": "microseconds", "order": "measurement order, unsorted", "stream": stream["description"],
        "native_add_us": native["add_order_us"], "native_cancel_us": native["cancel_order_us"],
        "binding_add_us": binding["add_us"], "binding_cancel_us": binding["cancel_us"],
        "binding_decimal_add_us": dec["add_us"], "binding_decimal_cancel_us": dec["cancel_us"],
        "python_engine_add_us": py["add_us"], "python_engine_cancel_us": py["cancel_us"]}), encoding="utf-8")
    section = {
        "stream": stream["description"], "orders": len(stream["ids"]), "cancels": len(stream["cancels"]),
        "native": {"add_order": native["add_order"], "cancel_order": native["cancel_order"], "passes": native["passes"],
                   "per_pass": native["per_pass"], "timer": native["timer"], "timer_overhead_us": native["timer_overhead_us"],
                   "far_levels_after_adds": native["far_levels_after_adds"], "window_ticks": native["window_ticks"]},
        "binding": {"add_order": stats(binding["add_us"]), "cancel_order": stats(binding["cancel_us"]),
                    "call": "exchange_engine._native.Book.add_limit_count / cancel, integer ticks and lots"},
        "binding_decimal": {"add_order": stats(dec["add_us"]), "cancel_order": stats(dec["cancel_us"]),
                            "call": "exchange_engine.native_book.NativeOrderBook.add_order / cancel_order with Decimal Orders"},
        "python_engine": {"add_order": stats(py["add_us"]), "cancel_order": stats(py["cancel_us"]),
                          "call": "exchange_engine.orderbook.OrderBook.add_order / cancel_order"},
        "counts": {k: {"fills": v[0], "resting_after_adds": v[1], "cancels_found": v[2]} for k, v in counts.items()},
        "consistent": len(set(counts.values())) == 1, "raw_file": raw_rel,
    }
    for name in ("binding", "binding_decimal", "python_engine"):
        section[f"p99_add_python_engine_over_{name}"] = (section["python_engine"]["add_order"]["p99_us"]
                                                          / section[name]["add_order"]["p99_us"])
    section["p99_add_python_engine_over_native"] = (section["python_engine"]["add_order"]["p99_us"]
                                                     / section["native"]["add_order"]["p99_us"])
    return section


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="C++ order book core latency, native and through the binding")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--orders", type=int, default=100_000)
    parser.add_argument("--cancels", type=int, default=10_000)
    parser.add_argument("--mid", type=float, default=64_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--build-dir", type=Path, default=REPO / "build" / "native")
    parser.add_argument("--no-build", action="store_true", help="use an existing build in --build-dir")
    parser.add_argument("--fuzz-streams", type=int, default=10_000, help="differential fuzz streams (0 = skip)")
    args = parser.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    build = {"skipped": True} if args.no_build else build_native(args.build_dir)
    stream = synthetic_stream(args.orders, args.cancels, args.mid, args.seed)
    section = measure(stream, args.results_dir, args.build_dir, args.passes)
    section.update({"build": build, "cpu_model": cpu_model(), "cpu_cores": os.cpu_count(), "python": platform.python_version(),
                    "platform": platform.platform()})
    if args.fuzz_streams:
        section["fuzz"] = fuzz(args.fuzz_streams)
        print(f"fuzz: {section['fuzz']['streams']:,} streams, {section['fuzz']['operations']:,} operations, "
              f"{section['fuzz']['divergences']} divergences", flush=True)
    metrics_path = args.results_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"source": "real", "sections": {}}
    metrics.setdefault("sections", {})["native_engine"] = section
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    for name in ("native", "binding", "binding_decimal", "python_engine"):
        a, c = section[name]["add_order"], section[name]["cancel_order"]
        print(f"{name:16s} add p50 {a['p50_us']:8.3f} p99 {a['p99_us']:8.3f} p99.9 {a['p999_us']:8.3f} us | "
              f"cancel p50 {c['p50_us']:8.3f} p99 {c['p99_us']:8.3f} p99.9 {c['p999_us']:8.3f} us")
    print(f"consistent across all four: {section['consistent']} {section['counts']}")
    return 0 if section["consistent"] and not (section.get("fuzz") or {}).get("divergences") else 1


if __name__ == "__main__":
    sys.exit(main())
