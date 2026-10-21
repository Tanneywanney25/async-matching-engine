"""Phase 4 precision study: float32 prefix sums and the rolling-window cancellation problem.

    python -m bench.bench_precision --archive <dir> --out <dir> --source synthetic|real [--quick]

Input: the top-of-book notional stream, ``float32(px_ticks) * size`` for every snapshot's best
bid followed by every snapshot's best ask, as float32 (over a million values for an archive of
500,000 snapshots or more). Window sums of 100 values (the VWAP numerator) are formed from prefix
sums four ways (``cuda/precision.cu``): float32 naive, float64, Kahan-compensated float32 chunks
with float64 carries, and block-rebased float32 with float64 carries. Each is compared with an
exact reference computed with Python integers (``bench.ref_scans.exact_prefix_and_window``), and
timed with CUDA events. Section ``precision``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from bench import ref_scans as ref
from bench.bench_scans import Scans
from bench.ddscan import DDScan
from bench.common import Bench, cupy, event_time_ms, load_archive, raw_module

KAHAN_CHUNK = 64
REBASE_BLOCK = 1024
BLOCK = 256
QUICK_N = 200_000


def log(*args) -> None:
    print("[precision]", *args, flush=True)


def build_input(arch: dict, quick: bool) -> np.ndarray:
    bid = arch["bid_px"][0].astype(np.float32) * arch["bid_sz"][0].astype(np.float32)
    ask = arch["ask_px"][0].astype(np.float32) * arch["ask_sz"][0].astype(np.float32)
    x = np.concatenate([bid, ask]).astype(np.float32)
    return x[:QUICK_N] if quick else x


def run(bench: Bench, arch: dict, quick: bool = False) -> None:
    cp = cupy()
    t0 = time.perf_counter()
    x = build_input(arch, quick)
    n = int(x.size)
    w = ref.PRECISION_WINDOW
    log(f"n={n:,} values, window={w}")
    t_exact = time.perf_counter()
    P_exact, W_exact, exact_meta = ref.exact_prefix_and_window(x, w)
    exact_seconds = time.perf_counter() - t_exact
    log(f"exact reference in {exact_seconds:.1f} s ({exact_meta})")

    sc = Scans()
    mod = raw_module(["precision.cu"])
    k = {name: mod.get_function(name) for name in ("kahan_chunks_f32", "block_local_scan_f32", "block_totals_f64",
                                                    "window_diff_local", "window_diff_f64")}
    stream = sc.stream
    d_x = cp.asarray(x)
    d_x64 = cp.asarray(x.astype(np.float64))
    grid = (max(1, math.ceil(n / BLOCK)),)
    n64, w32 = np.int64(n), np.int32(w)
    W = {name: cp.empty(n, cp.float64) for name in ("float32_naive", "float64", "kahan_float32", "block_rebased_float32",
                                                     "double_double")}
    P = {name: cp.empty(n, cp.float64) for name in W}
    L32 = cp.empty(n, cp.float32)
    C32 = cp.empty(n, cp.float32)
    L_reb = cp.empty(n, cp.float32)
    P64 = cp.empty(n, cp.float64)
    zero_off = cp.zeros(1, cp.float64)
    n_chunks = math.ceil(n / KAHAN_CHUNK)
    chunk_tot = cp.empty(n_chunks, cp.float64)
    n_blocks = math.ceil(n / REBASE_BLOCK)
    block_tot = cp.empty(n_blocks, cp.float64)
    L_kahan = cp.empty(n, cp.float32)
    dds = DDScan()
    D_hi, D_lo = cp.empty(n, cp.float64), cp.empty(n, cp.float64)

    def naive() -> None:
        sc.scan("warpscan", d_x, L32)
        k["window_diff_local"](grid, (BLOCK,), (L32, C32, zero_off, n64, np.int64(max(n, 1)), w32, np.int32(0),
                                                W["float32_naive"], P["float32_naive"]))

    def f64() -> None:
        sc.scan("warpscan", d_x64, P64)
        k["window_diff_f64"](grid, (BLOCK,), (P64, n64, w32, W["float64"]))

    def kahan() -> None:
        k["kahan_chunks_f32"]((max(1, math.ceil(n_chunks / BLOCK)),), (BLOCK,),
                              (d_x, L_kahan, C32, chunk_tot, n64, np.int32(KAHAN_CHUNK)))
        off = cp.cumsum(chunk_tot) - chunk_tot
        k["window_diff_local"](grid, (BLOCK,), (L_kahan, C32, off, n64, np.int64(KAHAN_CHUNK), w32, np.int32(1),
                                                W["kahan_float32"], P["kahan_float32"]))

    def rebased() -> None:
        k["block_local_scan_f32"]((n_blocks,), (BLOCK,), (d_x, L_reb, n64))
        k["block_totals_f64"]((n_blocks,), (BLOCK,), (d_x, block_tot, n64))
        off = cp.cumsum(block_tot) - block_tot
        k["window_diff_local"](grid, (BLOCK,), (L_reb, C32, off, n64, np.int64(REBASE_BLOCK), w32, np.int32(0),
                                                W["block_rebased_float32"], P["block_rebased_float32"]))

    def double_double() -> None:
        dds.scan(d_x64, D_hi, D_lo)
        dds.window(D_hi, D_lo, w, W["double_double"])

    fns = {"float32_naive": naive, "float64": f64, "kahan_float32": kahan, "block_rebased_float32": rebased,
           "double_double": double_double}
    for fn in fns.values():
        fn()
    stream.synchronize()
    P["float64"] = P64
    P["double_double"] = D_hi + D_lo
    errors = {}
    for name in fns:
        errors[name] = {"window": ref.rel_errors(cp.asnumpy(W[name]), W_exact),
                        "prefix": ref.rel_errors(cp.asnumpy(P[name]), P_exact)}
        log(f"{name}: window max {errors[name]['window']['max_rel_err']:.3e} "
            f"median {errors[name]['window']['median_rel_err']:.3e}; prefix max {errors[name]['prefix']['max_rel_err']:.3e}")
    timing = bench.block("precision", {name: (lambda fn=fn: event_time_ms(stream, fn)) for name, fn in fns.items()})

    max_px = float(max(arch["bid_px"][0].max(), arch["ask_px"][0].max()))
    tolerance = 0.1 / max_px  # a tenth of a tick at the largest price, relative
    variants = []
    for name in fns:
        variants.append({"name": name, "max_rel_err": errors[name]["window"]["max_rel_err"],
                         "median_rel_err": errors[name]["window"]["median_rel_err"],
                         "p99_rel_err": errors[name]["window"]["p99_rel_err"],
                         "window": errors[name]["window"], "prefix": errors[name]["prefix"], "ms": timing[name]})
    choice = ref.precision_choice(variants, tolerance)
    base_ms = timing["float32_naive"]["median_ms"]
    cost_pct = (choice["ms"]["median_ms"] - base_ms) / base_ms * 100.0
    within = [v["name"] for v in variants if v["window"]["max_rel_err"] <= tolerance]
    rationale = (f"Window sums of {w} values over {n:,} top-of-book notionals: float32 naive reaches a maximum relative "
                 f"error of {errors['float32_naive']['window']['max_rel_err']:.2e} because two large prefixes cancel; "
                 f"{choice['name']} keeps the maximum at {choice['window']['max_rel_err']:.2e}, inside the "
                 f"{tolerance:.1e} budget (a tenth of a tick at {max_px:,.0f} ticks), and is the fastest variant that "
                 f"does ({', '.join(within) or 'none'} qualify), at {cost_pct:+.1f}% time versus float32 naive.")
    bench.section("precision", {
        "n": n, "window": w, "variants": variants, "exact_reference": "python int over float32 mantissas in the base "
        "unit 2**min(exponent) (bench/ref_scans.py), rounded to float64 once", "exact_meta": exact_meta,
        "exact_seconds": exact_seconds, "input": "float32(px_ticks) * size for every best bid, then every best ask",
        "tolerance_rel": tolerance, "production_choice": choice["name"], "cost_vs_float32_naive_pct": cost_pct,
        "qualifying": within, "rationale": rationale, "kahan_chunk": KAHAN_CHUNK, "rebase_block": REBASE_BLOCK,
        "quick": bool(quick), "wall_seconds": time.perf_counter() - t0,
    })
    log(rationale)
    cp.get_default_memory_pool().free_all_blocks()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4: scan precision study")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", required=True, choices=("synthetic", "real"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    arch = load_archive(Path(args.archive))
    run(Bench(Path(args.out), args.source, arch["meta"]), arch, quick=args.quick)


if __name__ == "__main__":
    main()
