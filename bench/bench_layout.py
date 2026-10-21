"""Phase 4 layout profiling: AoS versus SoA book-level kernels under Nsight Systems.

Child mode (what ``nsys profile`` runs, exactly as in the build spec)::

    python bench/bench_layout.py --layout aos --archive <dir> [--iters 30]
    python bench/bench_layout.py --layout soa --archive <dir> [--iters 30]

Each runs warmup passes and then ``--iters`` passes of the four kernels in ``cuda/layout.cu``
(spread, depth, top-10 imbalance, per-snapshot cumulative depth) inside an NVTX range, and prints
one ``LAYOUT_RESULT {...}`` line with CUDA-event timings as a cross-check.

``run(bench, arch, quick)`` (used by ``bench.run_all``) profiles both layouts with
``bench.profile_layout`` and writes section ``layout``; bullet 2's figure is the reduction in total
kernel time parsed from the two ``cuda_gpu_kern_sum`` CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # run as a script under nsys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from bench.common import Bench, cupy, load_archive, raw_module, to_aos  # noqa: E402

BLOCK = 256
WARMUP = 3
DEFAULT_ITERS = 30
QUICK_ITERS = 5
QUICK_N = 100_000


def child(archive: Path, layout: str, iters: int, n_limit: int = 0) -> dict:
    cp = cupy()
    arch = load_archive(archive, names=("bid_px", "bid_sz", "ask_px", "ask_sz"))
    levels, n_total = arch["bid_px"].shape
    n = min(n_total, n_limit) if n_limit else n_total
    cols = {k: np.ascontiguousarray(arch[k][:, :n]) for k in ("bid_px", "bid_sz", "ask_px", "ask_sz")}
    mod = raw_module(["layout.cu"])
    names = ("spread", "depth", "imbalance", "cumdepth")
    k = {name: mod.get_function(f"layout_{name}_{layout}") for name in names}
    n64 = np.int64(n)
    grid = (max(1, math.ceil(n / BLOCK)),)
    spread = cp.empty(n, cp.int32)
    db, da, imb = cp.empty(n, cp.float32), cp.empty(n, cp.float32), cp.empty(n, cp.float32)
    cb, ca = cp.empty(n * levels, cp.float32), cp.empty(n * levels, cp.float32)
    if layout == "aos":
        recs = cp.asarray(to_aos(cols).view(np.uint8))
        launches = {
            "spread": lambda: k["spread"](grid, (BLOCK,), (recs, n64, spread)),
            "depth": lambda: k["depth"](grid, (BLOCK,), (recs, n64, db, da)),
            "imbalance": lambda: k["imbalance"](grid, (BLOCK,), (recs, n64, imb)),
            "cumdepth": lambda: k["cumdepth"](grid, (BLOCK,), (recs, n64, cb, ca)),
        }
    else:
        d = {key: cp.asarray(v) for key, v in cols.items()}
        launches = {
            "spread": lambda: k["spread"](grid, (BLOCK,), (d["bid_px"], d["ask_px"], n64, spread)),
            "depth": lambda: k["depth"](grid, (BLOCK,), (d["bid_sz"], d["ask_sz"], n64, db, da)),
            "imbalance": lambda: k["imbalance"](grid, (BLOCK,), (d["bid_sz"], d["ask_sz"], n64, imb)),
            "cumdepth": lambda: k["cumdepth"](grid, (BLOCK,), (d["bid_sz"], d["ask_sz"], n64, cb, ca)),
        }
    for _ in range(WARMUP):
        for fn in launches.values():
            fn()
    cp.cuda.Device().synchronize()
    nvtx = cp.cuda.nvtx
    stream = cp.cuda.get_current_stream()
    ev = {name: [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(iters)] for name in names}
    nvtx.RangePush(f"layout_{layout}")
    for it in range(iters):
        for name in names:
            start, end = ev[name][it]
            start.record(stream)
            launches[name]()
            end.record(stream)
    nvtx.RangePop()
    cp.cuda.Device().synchronize()
    per = {name: [cp.cuda.get_elapsed_time(s, e) for s, e in ev[name]] for name in names}
    # spot checks so a broken kernel cannot produce a flattering profile
    want_spread = cols["ask_px"][0].astype(np.int64) - cols["bid_px"][0].astype(np.int64)
    ok = bool(np.array_equal(cp.asnumpy(spread).astype(np.int64), want_spread))
    want_depth = cols["bid_sz"].sum(axis=0, dtype=np.float64)
    ok = ok and bool(np.allclose(cp.asnumpy(db), want_depth, rtol=1e-5, atol=1e-4))
    last = cp.asnumpy(cb[(n - 1) * levels + levels - 1: (n - 1) * levels + levels] if layout == "aos"
                      else cb[(levels - 1) * n + n - 1: (levels - 1) * n + n])
    ok = ok and bool(np.allclose(last, want_depth[-1:], rtol=1e-5, atol=1e-4))
    return {"layout": layout, "n": n, "levels": int(levels), "iters": iters, "verified": ok,
            "event_median_ms": {name: float(np.median(v)) for name, v in per.items()},
            "event_total_median_ms": float(sum(np.median(v) for v in per.values()))}


def run(bench: Bench, arch: dict, quick: bool = False) -> None:
    from bench import profile_layout

    t0 = time.perf_counter()
    section = profile_layout.profile(bench.out_dir, Path(arch["path"]), iters=QUICK_ITERS if quick else DEFAULT_ITERS,
                                     n_limit=QUICK_N if quick else 0)
    section["quick"] = bool(quick)
    section["wall_seconds"] = time.perf_counter() - t0
    bench.section("layout", section)


def main() -> None:
    parser = argparse.ArgumentParser(description="AoS vs SoA book-level kernels (nsys child)")
    parser.add_argument("--layout", required=True, choices=("aos", "soa"))
    parser.add_argument("--archive", required=True)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--n", type=int, default=0, help="use the first N snapshots (0 = all)")
    args = parser.parse_args()
    res = child(Path(args.archive), args.layout, args.iters, args.n)
    print("LAYOUT_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
