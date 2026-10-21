"""Run every benchmark section in one job and leave a complete results manifest.

    python -m bench.run_all --archive <dir> --out <dir> --source real|synthetic
                            [--quick] [--sections ladder,scans,...|none] [--iters N]
                            [--engine-orders N] [--engine-cancels N] [--skip-engine]
                            [--budget-minutes M]

Sections run in this order, each through ``bench.bench_<name>.run(bench, arch, quick)``:
``ladder, scans, precision, layout, slippage, ofi, pipeline``; then the matching-engine
latency benchmark runs as a subprocess on the host CPU. A section that raises
(``ImportError`` for a module that does not exist yet included) is recorded in
``metrics["errors"]`` as ``{"section", "error"}`` and the job moves on, and
``metrics.json`` is rewritten after every section, so a job that dies in one kernel
still leaves every other measurement behind.

``--sections`` picks the GPU sections (``none`` runs no GPU section at all); the engine
step is independent and runs unless ``--skip-engine`` is given. ``--quick`` is passed to
every section and lowers the default iteration count so a smoke job finishes in about a
minute. The process exits 0 even when sections failed (the manifest says so) and
non-zero only when the archive itself cannot be loaded.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from bench.common import MIN_ITERS, REPO_ROOT, Bench, load_archive

GPU_SECTIONS: Tuple[str, ...] = ("ladder", "scans", "precision", "layout", "slippage", "ofi", "pipeline")
QUICK_ITERS = 5
ENGINE_RAW_NAME = "engine_latency_us.json"
EXIT_ARCHIVE = 2


# --------------------------------------------------------------------------- #
# Device facts
# --------------------------------------------------------------------------- #


def gpu_block() -> dict:
    """``metrics["gpu"]``: read from the card through CuPy, never assumed."""
    block: dict = {"name": None, "sm_count": None, "memory_gb": None,
                   "compute_capability": None, "theoretical_bw_gbs": None}
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        block.update({
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "sm_count": int(props["multiProcessorCount"]),
            "memory_gb": round(int(props["totalGlobalMem"]) / 1e9, 3),
            "compute_capability": f"{props['major']}.{props['minor']}",
            # memoryClockRate is in kHz; DDR moves two words per clock.
            "theoretical_bw_gbs": round(2 * props["memoryClockRate"] * 1e3 * props["memoryBusWidth"] / 8 / 1e9, 2),
            "memory_clock_khz": int(props["memoryClockRate"]),
            "memory_bus_width_bits": int(props["memoryBusWidth"]),
            "shared_mem_per_sm_bytes": int(props.get("sharedMemPerMultiprocessor", 0)),
            "l2_cache_bytes": int(props.get("l2CacheSize", 0)),
            "cupy_version": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "nvrtc_arch": "sm_" + cp.cuda.Device().compute_capability,
        })
    except Exception as exc:  # noqa: BLE001 - no CuPy or no card: record it and carry on
        block["error"] = repr(exc)
        try:
            line = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            name, mem = [x.strip() for x in line.split(",")[:2]]
            block["name"] = name
            block["memory_gb"] = round(float(mem) * 1048576 / 1e9, 3)
        except Exception:  # noqa: BLE001
            pass
    return block


def release_gpu_memory() -> None:
    """Hand pooled device memory back between sections so one section's working set
    cannot starve the next on a 15 GB card."""
    cp = sys.modules.get("cupy")
    if cp is None:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def parse_sections(text: Optional[str]) -> List[str]:
    """``None``/``all`` -> every GPU section, ``none`` -> no GPU section, else a comma list."""
    if text is None or text.strip().lower() == "all":
        return list(GPU_SECTIONS)
    if text.strip().lower() == "none":
        return []
    names = [t.strip().lower() for t in text.split(",") if t.strip()]
    unknown = [n for n in names if n not in GPU_SECTIONS]
    if unknown:
        raise ValueError(f"unknown section(s) {unknown}; choose from {', '.join(GPU_SECTIONS)} "
                         "(the engine step is controlled by --skip-engine)")
    return names


def _record_error(errors: List[dict], name: str, exc: BaseException, t0: float) -> None:
    errors.append({"section": name, "type": type(exc).__name__, "error": traceback.format_exc(),
                   "seconds": round(time.perf_counter() - t0, 3)})


def run_section(bench: Bench, arch: dict, name: str, quick: bool, errors: List[dict]) -> Tuple[bool, float]:
    """Import ``bench.bench_<name>`` and call ``run(bench, arch, quick)``.

    Any exception, a missing module included, lands in ``errors`` with the traceback;
    the timing goes to ``metrics["timings_s"]`` and ``metrics.json`` is saved either way.
    """
    t0 = time.perf_counter()
    ok = True
    try:
        module = importlib.import_module(f"bench.bench_{name}")
        module.run(bench, arch, quick)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - one bad section must not sink the job
        ok = False
        _record_error(errors, name, exc, t0)
    seconds = time.perf_counter() - t0
    bench.metrics.setdefault("timings_s", {})[name] = round(seconds, 3)
    bench.save()
    return ok, seconds


# --------------------------------------------------------------------------- #
# Engine benchmark (host CPU, subprocess)
# --------------------------------------------------------------------------- #


def _engine_section_from(engine_dir: Path) -> Optional[dict]:
    """Read ``sections.engine`` out of ``<engine_dir>/metrics.json`` if it is there."""
    path = engine_dir / "metrics.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    section = None
    if isinstance(data.get("sections"), dict):
        section = data["sections"].get("engine")
    if section is None and isinstance(data.get("engine"), dict):
        section = data["engine"]
    if section is None and "add_order" in data:
        section = data
    return dict(section) if isinstance(section, dict) else None


def _find_engine_raw(engine_dir: Path, raw_file: Optional[str]) -> Optional[Path]:
    if raw_file:
        candidate = engine_dir / raw_file
        if candidate.exists():
            return candidate
    return next(engine_dir.rglob(ENGINE_RAW_NAME), None)


def _print_block(text: str) -> None:
    text = text.strip()
    if text:
        print(textwrap.indent(text, "    "), flush=True)


def engine_benchmark(engine_dir: Path, raw_dir: Path, orders: int, cancels: int) -> dict:
    """Run ``scripts/benchmark.py --results-dir`` as a subprocess and return the engine section.

    If that script cannot write a results file yet (an older copy without
    ``--results-dir``), the same measurement runs through ``bench.run_all
    --engine-standalone`` in its own interpreter, so the section always exists and
    always carries the contract shape. The raw latency arrays are copied to
    ``<out>/raw/engine_latency_us.json`` and ``raw_file`` points there.
    """
    engine_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "scripts" / "benchmark.py"
    cmd = [sys.executable, str(script), "--orders", str(orders), "--cancels", str(cancels),
           "--results-dir", str(engine_dir)]
    print("[run_all] $ python " + " ".join(cmd[1:]), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600)
    _print_block(proc.stdout)
    section = _engine_section_from(engine_dir) if proc.returncode == 0 else None
    if section is None:
        err_lines = proc.stderr.strip().splitlines()
        reason = err_lines[-1] if err_lines else f"exit {proc.returncode} without a results file"
        print(f"[run_all] scripts/benchmark.py left no results file ({reason[:200]}); "
              "running the built-in equivalent", flush=True)
        cmd = [sys.executable, "-m", "bench.run_all", "--engine-standalone", "--out", str(engine_dir),
               "--engine-orders", str(orders), "--engine-cancels", str(cancels)]
        print("[run_all] $ python " + " ".join(cmd[1:]), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600)
        _print_block(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"engine fallback failed ({proc.returncode}): {proc.stderr[-2000:]}")
        section = _engine_section_from(engine_dir)
        if section is None:
            raise RuntimeError("engine fallback produced no metrics.json")
        section["runner"] = "bench.run_all --engine-standalone"
        section["script_note"] = f"scripts/benchmark.py --results-dir unavailable: {reason[:300]}"
    else:
        section.setdefault("runner", "scripts/benchmark.py --results-dir")
    raw = _find_engine_raw(engine_dir, section.get("raw_file"))
    if raw is not None:
        target = raw_dir / ENGINE_RAW_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        if raw.resolve() != target.resolve():
            shutil.copyfile(raw, target)
        section["raw_file"] = f"raw/{ENGINE_RAW_NAME}"
    else:
        section["raw_file_missing"] = True
    section.setdefault("orders", orders)
    section.setdefault("cancels", cancels)
    return section


def run_engine(bench: Bench, orders: int, cancels: int, errors: List[dict]) -> Tuple[bool, float]:
    t0 = time.perf_counter()
    ok = True
    try:
        section = engine_benchmark(bench.out_dir / "engine", bench.raw_dir, orders, cancels)
        bench.section("engine", section)
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        ok = False
        _record_error(errors, "engine", exc, t0)
    seconds = time.perf_counter() - t0
    bench.metrics.setdefault("timings_s", {})["engine"] = round(seconds, 3)
    bench.save()
    return ok, seconds


def _percentile(sorted_us: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile, identical to ``scripts/benchmark.py``."""
    if not sorted_us:
        return 0.0
    rank = max(0, min(len(sorted_us) - 1, int(round(pct / 100 * len(sorted_us))) - 1))
    return float(sorted_us[rank])


def latency_stats(samples_us: Sequence[float]) -> dict:
    xs = sorted(float(x) for x in samples_us)
    return {"mean_us": (sum(xs) / len(xs)) if xs else 0.0, "p50_us": _percentile(xs, 50),
            "p95_us": _percentile(xs, 95), "p99_us": _percentile(xs, 99), "n": len(xs)}


def cpu_model() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def engine_standalone(out_dir: Path, orders: int, cancels: int, mid: float = 64_000.0, seed: int = 42) -> int:
    """The measurement of ``scripts/benchmark.py`` (same seed, same order stream, same
    nearest-rank percentiles), written as ``metrics.json`` plus the raw latency arrays."""
    import random
    from decimal import Decimal

    from exchange_engine.models import Order, Side
    from exchange_engine.orderbook import OrderBook

    rng = random.Random(seed)
    book = OrderBook("BENCH")
    add_us: List[float] = []
    resting: List[int] = []
    for _ in range(orders):
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        price = Decimal(f"{rng.gauss(mid, mid * 0.01):.2f}")
        quantity = Decimal(f"{rng.uniform(0.001, 2.0):.6f}")
        order = Order(side, price, quantity)
        t0 = time.perf_counter_ns()
        book.add_order(order)
        add_us.append((time.perf_counter_ns() - t0) / 1000.0)
        if order.quantity > 0:
            resting.append(order.id)
    shuffler = random.Random(seed + 1)
    shuffler.shuffle(resting)
    cancel_us: List[float] = []
    for order_id in resting[:cancels]:
        t0 = time.perf_counter_ns()
        book.cancel_order(order_id)
        cancel_us.append((time.perf_counter_ns() - t0) / 1000.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / ENGINE_RAW_NAME).write_text(json.dumps({
        "orders": orders, "cancels": len(cancel_us), "mid": mid, "seed": seed, "unit": "microseconds",
        "add_order_us": add_us, "cancel_order_us": cancel_us}), encoding="utf-8")
    section = {
        "orders": orders, "cancels": len(cancel_us), "python": platform.python_version(),
        "cpu_model": cpu_model(), "platform": platform.platform(), "mid": mid, "seed": seed,
        "add_order": latency_stats(add_us), "cancel_order": latency_stats(cancel_us),
        "raw_file": f"raw/{ENGINE_RAW_NAME}", "runner": "bench.run_all --engine-standalone",
    }
    (out_dir / "metrics.json").write_text(json.dumps({"sections": {"engine": section}}, indent=2), encoding="utf-8")
    print(f"add_order: p50 {section['add_order']['p50_us']:.2f} us, p99 {section['add_order']['p99_us']:.2f} us "
          f"over {orders:,} orders; cancel_order: p99 {section['cancel_order']['p99_us']:.2f} us "
          f"over {len(cancel_us):,} cancels", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bench.run_all",
                                description="Run every benchmark section and the engine benchmark into one results directory.")
    p.add_argument("--archive", help="archive directory (archive.json plus the arrays)")
    p.add_argument("--out", required=True, help="results directory (metrics.json, raw/)")
    p.add_argument("--source", choices=["real", "synthetic"], help="provenance label; must match the archive")
    p.add_argument("--quick", action="store_true", help="few iterations on a subset (smoke run)")
    p.add_argument("--sections", default=None,
                   help=f"comma list of GPU sections from {','.join(GPU_SECTIONS)}, or 'none' (default: all)")
    p.add_argument("--iters", type=int, default=None,
                   help=f"timed iterations per block (default {MIN_ITERS}, {QUICK_ITERS} with --quick)")
    p.add_argument("--engine-orders", type=int, default=100_000)
    p.add_argument("--engine-cancels", type=int, default=10_000)
    p.add_argument("--skip-engine", action="store_true", help="do not run scripts/benchmark.py")
    p.add_argument("--budget-minutes", type=float, default=0.0,
                   help="skip GPU sections not started within this wall-clock budget (0 = no limit)")
    p.add_argument("--engine-standalone", action="store_true", help=argparse.SUPPRESS)
    return p


def _report(name: str, ok: bool, seconds: float, errors: List[dict]) -> None:
    if ok:
        print(f"[run_all] {name:<10} {seconds:8.1f} s  ok", flush=True)
        return
    last = ""
    if errors:
        lines = (errors[-1].get("error") or "").strip().splitlines()
        last = lines[-1] if lines else errors[-1].get("type", "")
    print(f"[run_all] {name:<10} {seconds:8.1f} s  FAILED  {last[:160]}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.engine_standalone:
        return engine_standalone(Path(args.out), args.engine_orders, args.engine_cancels)
    if not args.archive or not args.source:
        parser.error("--archive and --source are required")
    try:
        sections = parse_sections(args.sections)
    except ValueError as exc:
        parser.error(str(exc))

    t_start = time.time()
    archive_dir = Path(args.archive)
    try:
        arch = load_archive(archive_dir)
    except Exception as exc:  # noqa: BLE001 - the one fatal condition
        print(f"[run_all] cannot load archive {archive_dir}: {exc!r}", file=sys.stderr, flush=True)
        return EXIT_ARCHIVE
    n_iters = args.iters if args.iters else (QUICK_ITERS if args.quick else MIN_ITERS)
    try:
        bench = Bench(Path(args.out), args.source, arch["meta"], n_iters=n_iters)
    except ValueError as exc:  # archive source does not match --source
        print(f"[run_all] {exc}", file=sys.stderr, flush=True)
        return EXIT_ARCHIVE

    bench.metrics["gpu"] = gpu_block()
    errors: List[dict] = bench.metrics.setdefault("errors", [])
    rerun = set(sections) | (set() if args.skip_engine else {"engine"})
    errors[:] = [e for e in errors if e.get("section") not in rerun]
    bench.metrics["run_all"] = {
        "argv": list(argv) if argv is not None else sys.argv[1:], "quick": bool(args.quick),
        "n_iters": bench.n_iters, "sections_requested": sections, "engine": not args.skip_engine,
        "started_unix": t_start, "python": platform.python_version(), "platform": platform.platform(),
        "hostname": platform.node(),
    }
    bench.save()

    meta = arch["meta"]
    print(f"[run_all] archive {archive_dir} ({meta.get('source')}, {int(meta.get('n_snapshots') or 0):,} snapshots, "
          f"{int(meta.get('n_trades') or 0):,} trades) -> {bench.out_dir}", flush=True)
    engine_note = "skipped" if args.skip_engine else f"{args.engine_orders:,} orders / {args.engine_cancels:,} cancels"
    print(f"[run_all] gpu {bench.metrics['gpu'].get('name') or 'none'}; sections {', '.join(sections) or 'none'}; "
          f"engine {engine_note}; iters {bench.n_iters}; quick {args.quick}", flush=True)

    budget_s = args.budget_minutes * 60.0
    for name in sections:
        elapsed = time.time() - t_start
        if budget_s and elapsed > budget_s:
            errors.append({"section": name, "type": "Skipped", "seconds": 0.0,
                           "error": f"skipped: {elapsed / 60:.1f} min elapsed exceeds --budget-minutes {args.budget_minutes:g}"})
            bench.save()
            print(f"[run_all] {name:<10} {'':>8}    skipped (time budget)", flush=True)
            continue
        ok, seconds = run_section(bench, arch, name, args.quick, errors)
        _report(name, ok, seconds, errors)
        release_gpu_memory()
    if not args.skip_engine:
        ok, seconds = run_engine(bench, args.engine_orders, args.engine_cancels, errors)
        _report("engine", ok, seconds, errors)

    bench.metrics["run_all"]["finished_unix"] = time.time()
    bench.metrics["run_all"]["wall_seconds"] = round(time.time() - t_start, 1)
    bench.save()
    print(f"[run_all] done in {time.time() - t_start:.1f} s: {len(bench.metrics['sections'])} section(s) in "
          f"{bench.path}, {len(errors)} error(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
