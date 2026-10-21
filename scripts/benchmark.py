"""Latency benchmark for the :class:`~exchange_engine.orderbook.OrderBook`.

Generates a large batch of random limit orders around a mid-price and measures
per-operation insertion+match latency, then measures cancel latency. Reports
p50/p95/p99 in microseconds.

Run with::

    python scripts/benchmark.py [--orders 100000] [--cancels 10000]

With ``--results-dir <dir>`` the same run also writes, next to the unchanged printout,
the files the documentation renderer reads (shape: ``bench/CONTRACT.md``, section
``engine``): ``<dir>/metrics.json``, ``<dir>/manifest.json`` and the full latency
arrays in ``<dir>/raw/engine_latency_us.json``. The figures cover the matching path
only: ``add_order`` and ``cancel_order`` on an in-memory book, no feed, no network.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

# Allow running directly from a checkout without installing the package.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exchange_engine.models import Order, Side  # noqa: E402
from exchange_engine.orderbook import OrderBook  # noqa: E402


def _percentile(sorted_us: List[float], pct: float) -> float:
    """Return the ``pct`` percentile from an already-sorted list (nearest-rank)."""
    if not sorted_us:
        return 0.0
    rank = max(0, min(len(sorted_us) - 1, int(round(pct / 100 * len(sorted_us))) - 1))
    return sorted_us[rank]


def _report(name: str, latencies_us: List[float]) -> None:
    """Print p50/p95/p99 latency statistics for a labelled operation."""
    latencies_us.sort()
    p50 = _percentile(latencies_us, 50)
    p95 = _percentile(latencies_us, 95)
    p99 = _percentile(latencies_us, 99)
    mean = sum(latencies_us) / len(latencies_us)
    print(f"\n{name} ({len(latencies_us):,} ops)")
    print(f"  mean : {mean:8.2f} us")
    print(f"  p50  : {p50:8.2f} us")
    print(f"  p95  : {p95:8.2f} us")
    print(f"  p99  : {p99:8.2f} us")


def _stats(latencies_us: List[float]) -> dict:
    """mean/p50/p95/p99 in microseconds, the same nearest-rank percentiles as the printout."""
    xs = sorted(latencies_us)
    if not xs:
        return {"mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0}
    return {"mean_us": sum(xs) / len(xs), "p50_us": _percentile(xs, 50),
            "p95_us": _percentile(xs, 95), "p99_us": _percentile(xs, 99)}


def _cpu_model() -> str:
    """The CPU's marketing name from the first source that answers: the Windows registry
    (``wmic`` is absent from recent Windows 11 builds), ``wmic``, ``/proc/cpuinfo``,
    ``sysctl`` on macOS, and finally :mod:`platform`."""

    def registry() -> str:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0])

    def wmic() -> str:
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True,
                             timeout=10).stdout
        return next(ln.strip() for ln in out.splitlines()
                    if ln.strip() and ln.strip().lower() != "name")

    def proc_cpuinfo() -> str:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        return next(ln.split(":", 1)[1] for ln in text.splitlines()
                    if ln.lower().startswith("model name"))

    def sysctl() -> str:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=10).stdout

    sources = [registry, wmic] if sys.platform == "win32" else [proc_cpuinfo, sysctl]
    for source in sources:
        try:
            name = " ".join(source().split())
            if name:
                return name
        except Exception:  # noqa: BLE001 - fall through to the next source
            continue
    return platform.processor() or platform.machine() or "unknown"


def _git_sha() -> Optional[str]:
    """Commit of the checkout being benchmarked (``None`` when git is unavailable)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).resolve().parent.parent)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def write_results(results_dir: Path, args: argparse.Namespace, add_us: List[float],
                  cancel_us: List[float], started_unix: float, finished_unix: float) -> None:
    """Write ``metrics.json`` (section ``engine``), ``manifest.json`` and the raw latency arrays.

    An existing ``metrics.json`` (for example a GPU job's, written incrementally by
    ``bench.common.Bench``) only gains the ``engine`` section and keeps its own ``source``;
    an existing ``manifest.json`` belongs to whoever wrote it and is left untouched.
    """
    results_dir = Path(results_dir)
    raw_rel = "raw/engine_latency_us.json"
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)
    cpu = _cpu_model()
    (results_dir / raw_rel).write_text(json.dumps({
        "unit": "microseconds", "order": "measurement order, unsorted",
        "orders": len(add_us), "cancels": len(cancel_us), "mid": args.mid, "seed": args.seed,
        "add_order_us": add_us, "cancel_order_us": cancel_us}), encoding="utf-8")

    engine = {"orders": len(add_us), "cancels": len(cancel_us), "python": platform.python_version(),
              "cpu_model": cpu, "add_order": _stats(add_us), "cancel_order": _stats(cancel_us),
              "raw_file": raw_rel}
    metrics_path = results_dir / "metrics.json"
    metrics: dict = {"source": "real", "sections": {}}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("sections", {})
    metrics["sections"]["engine"] = engine
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        print(f"\nLeft the existing {manifest_path} untouched.")
    else:
        manifest = {"run_id": results_dir.resolve().name, "job": "engine_local", "source": "real",
                    "gpu_name": "none (CPU benchmark)", "cpu_model": cpu, "cpu_cores": os.cpu_count(),
                    "python": platform.python_version(), "os": platform.platform(),
                    "git_sha": _git_sha(), "started_unix": started_unix,
                    "finished_unix": finished_unix, "wall_seconds": finished_unix - started_unix,
                    "status": "complete"}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path}, {manifest_path} and {results_dir / raw_rel}")


def benchmark_inserts(n_orders: int, mid: float, seed: int,
                      latencies_out: Optional[List[float]] = None) -> "tuple[OrderBook, List[int]]":
    """Insert ``n_orders`` random limit orders, timing each add_order call.

    Returns the populated book and the list of resting order ids (for cancels).
    ``latencies_out``, when given, receives every add_order latency in measurement order.
    """
    rng = random.Random(seed)
    book = OrderBook("BENCH")
    latencies_us: List[float] = []
    resting_ids: List[int] = []

    for _ in range(n_orders):
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        price = Decimal(f"{rng.gauss(mid, mid * 0.01):.2f}")
        quantity = Decimal(f"{rng.uniform(0.001, 2.0):.6f}")
        order = Order(side, price, quantity)

        start = time.perf_counter_ns()
        book.add_order(order)
        latencies_us.append((time.perf_counter_ns() - start) / 1000.0)

        if order.quantity > 0:  # order rested (fully or partially)
            resting_ids.append(order.id)

    if latencies_out is not None:
        latencies_out.extend(latencies_us)  # before _report sorts the list in place
    _report("add_order (insert + match)", latencies_us)
    return book, resting_ids


def main() -> None:
    """Parse arguments and run the insertion and cancel benchmarks."""
    parser = argparse.ArgumentParser(description="OrderBook latency benchmark")
    parser.add_argument("--orders", type=int, default=100_000)
    parser.add_argument("--cancels", type=int, default=10_000)
    parser.add_argument("--mid", type=float, default=64_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="also write metrics.json, manifest.json and raw/engine_latency_us.json "
                             "into this directory (see bench/CONTRACT.md, section engine)")
    args = parser.parse_args()
    started_unix = time.time()

    print(f"Benchmarking OrderBook with {args.orders:,} orders (mid={args.mid:,.0f})...")
    add_us: List[float] = []
    book, resting_ids = benchmark_inserts(args.orders, args.mid, args.seed, latencies_out=add_us)

    rng = random.Random(args.seed + 1)
    rng.shuffle(resting_ids)
    to_cancel = resting_ids[: args.cancels]
    cancel_us: List[float] = []
    for order_id in to_cancel:
        start = time.perf_counter_ns()
        book.cancel_order(order_id)
        cancel_us.append((time.perf_counter_ns() - start) / 1000.0)
    finished_unix = time.time()
    cancel_raw = list(cancel_us)  # before _report sorts the list in place
    if cancel_us:
        _report("cancel_order", cancel_us)

    print("\nTarget: median add_order under 50 us.")

    if args.results_dir is not None:
        write_results(args.results_dir, args, add_us, cancel_raw, started_unix, finished_unix)


if __name__ == "__main__":
    main()
