"""Correctness and liveness check for the decoupled look-back scan, run in its own process.

    python -m bench.lookback_check [--quick]

``bench.bench_scans`` runs this with a timeout before timing anything: a look-back scan that
deadlocks hangs its kernel, and in a child process that only costs the timeout (killing the
process tears down its CUDA context), where in the benchmark process it would hold the GPU job
until Kaggle's own limit. Prints one ``LOOKBACK_CHECK {json}`` line and exits 0 when every case
passed.

Cases: sizes around the tile boundary, the archive's length and 2**24; integer-valued inputs,
whose partial sums are exact in float64, so the result must equal ``numpy.cumsum`` bit for bit
whatever order the look-back added the tiles in (a race shows up as a wrong integer, not as
rounding); real-valued inputs within 1e-9 relative; and repeated runs at 2**24 to catch races
that only bite sometimes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

TILE = 256 * 8
SIZES = (1, 7, TILE - 1, TILE, TILE + 1, 3 * TILE + 5, 863_153, 1 << 24)
QUICK_SIZES = (1, 7, TILE - 1, TILE, TILE + 1, 3 * TILE + 5, 100_000)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    from bench.bench_scans import Scans

    import cupy as cp

    sc = Scans(with_cub=False)
    rng = np.random.default_rng(7)
    cases, ok = [], True
    t0 = time.perf_counter()
    for n in (QUICK_SIZES if args.quick else SIZES):
        for kind in ("integer", "real"):
            host = rng.integers(-1000, 1000, n).astype(np.float64) if kind == "integer" else rng.standard_normal(n) * 1e3
            x, out = cp.asarray(host), cp.full(n, np.nan)
            sc.scan("lookback", x, out)
            got, want = cp.asnumpy(out), np.cumsum(host)
            if kind == "integer":
                passed = bool(np.array_equal(got, want))
                err = float(np.max(np.abs(got - want)))
            else:
                err = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-300)))
                passed = err <= 1e-9 and not np.isnan(got).any()
            cases.append({"n": n, "input": kind, "passed": passed, "max_err": err})
            ok = ok and passed
    n = 100_000 if args.quick else 1 << 24
    host = rng.integers(-1000, 1000, n).astype(np.float64)
    want = np.cumsum(host)
    x, out = cp.asarray(host), cp.empty(n)
    failures = 0
    for _ in range(args.repeats):
        out.fill(np.nan)
        sc.scan("lookback", x, out)
        failures += int(not np.array_equal(cp.asnumpy(out), want))
    ok = ok and failures == 0
    report = {"passed": ok, "cases": cases, "repeat_n": n, "repeats": args.repeats, "repeat_failures": failures,
              "seconds": round(time.perf_counter() - t0, 2)}
    print("LOOKBACK_CHECK " + json.dumps(report), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
