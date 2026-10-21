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
rounding); zero-mean real-valued inputs, compared against ``numpy.cumsum`` relative to the
running sum of absolute values (the scale floating-point summation error is bounded by; the
running sum itself crosses zero, where any change of summation order looks like a huge relative
error); and repeated runs at 2**24 to catch races that only bite sometimes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

TILE = 256 * 8
REAL_TOLERANCE = 1e-12  # relative to the running sum of |x|; observed differences are far below it


def compare(kind: str, host: np.ndarray, got: np.ndarray) -> tuple:
    """``(passed, max error)`` of a scan result against ``numpy.cumsum``.

    Integer-valued inputs must match exactly. Real-valued ones are measured against the running
    sum of absolute values, which bounds the rounding of any summation order.
    """
    want = np.cumsum(host)
    if kind == "integer":
        return bool(np.array_equal(got, want)), float(np.max(np.abs(got - want))) if host.size else 0.0
    scale = np.maximum(np.cumsum(np.abs(host)), 1e-300)
    err = float(np.max(np.abs(got - want) / scale)) if host.size else 0.0
    return bool(err <= REAL_TOLERANCE and not np.isnan(got).any()), err


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
            passed, err = compare(kind, host, cp.asnumpy(out))
            cases.append({"n": n, "input": kind, "passed": passed, "max_err": err,
                          "measure": "exact" if kind == "integer" else "relative to the running sum of |x|"})
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
