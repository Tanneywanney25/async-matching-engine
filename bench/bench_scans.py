"""Phase 4 benchmark: prefix scans and the rolling windows built on them.

    python -m bench.bench_scans --archive <dir> --out <dir> --source synthetic|real [--quick]

Section ``scans`` (shape in ``bench/CONTRACT.md``):

* three inclusive float64 scans, interleaved, CUDA events: the work-efficient Blelloch scan in
  shared memory with bank-conflict padding, the warp-shuffle scan, and ``cub::DeviceScan::
  InclusiveSum`` (built in-job with nvcc); at the archive's snapshot count and, for throughput,
  at 2**24 elements (the real series tiled). Each is checked against ``numpy.cumsum``.
* rolling VWAP over trades as two inclusive scans (price x size, size) per product plus a windowed
  difference: the 100-trade window of ``metrics.rolling_vwap`` (checked in Decimal on 200 sampled
  trades) and 1 s / 10 s windows sampled at every snapshot (checked against NumPy).
* rolling 1 s mean spread and top-10 imbalance over the snapshot stream with the same machinery.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from bench import ref_scans as ref
from bench.common import REPO_ROOT, Bench, cupy, event_time_ms, load_archive, raw_module
from bench.ddscan import DDScan

SCAN_ELEMS = 1024
LARGE_N = 1 << 24
QUICK_LARGE_N = 1 << 20
QUICK_N = 100_000
VWAP_SAMPLE = 200
BLOCK = 256


def log(*args) -> None:
    print("[scans]", *args, flush=True)


class Scans:
    """The NVRTC scans (multi-block, recursive carries) plus the CUB library when it builds."""

    def __init__(self) -> None:
        cp = self.cp = cupy()
        self.mod = raw_module(["scan.cu"])
        self.k = {name: self.mod.get_function(name) for name in (
            "blelloch_f64", "blelloch_f32", "warpscan_f64", "warpscan_f32",
            "scan_add_offsets_f64", "scan_add_offsets_f32")}
        self.bufs: Dict[tuple, list] = {}
        self.cub = None
        self.cub_error: Optional[str] = None
        try:
            self.cub = build_cub()
        except Exception as exc:  # noqa: BLE001 - recorded, CUB skipped honestly
            self.cub_error = repr(exc)
            log("CUB unavailable:", exc)
        self.stream = cp.cuda.get_current_stream()

    def _levels(self, n: int, dtype) -> list:
        key = (n, np.dtype(dtype).str)
        if key not in self.bufs:
            levels, m = [], n
            while True:
                nb = math.ceil(m / SCAN_ELEMS)
                levels.append((m, self.cp.empty(nb, dtype=dtype), self.cp.empty(nb, dtype=dtype)))
                if nb <= 1:
                    break
                m = nb
            self.bufs[key] = levels
        return self.bufs[key]

    def scan(self, kind: str, x, out) -> None:
        """Inclusive scan of ``x`` into ``out`` with ``kind`` in blelloch / warpscan / cub."""
        if kind == "cub":
            fn = self.cub.cub_inclusive_sum_f64 if x.dtype == np.float64 else self.cub.cub_inclusive_sum_f32
            rc = fn(ctypes.c_void_p(x.data.ptr), ctypes.c_void_p(out.data.ptr), ctypes.c_longlong(x.size),
                    ctypes.c_void_p(self.stream.ptr))
            if rc != 0:
                raise RuntimeError(f"cub returned {rc}")
            return
        suffix = "f64" if x.dtype == np.float64 else "f32"
        self._scan(kind, suffix, x, out, self._levels(int(x.size), x.dtype), 0)

    def _scan(self, kind, suffix, x, out, levels, depth) -> None:
        m, sums, scanned = levels[depth]
        nb = sums.size
        threads = 512 if kind == "blelloch" else 256
        self.k[f"{kind}_{suffix}"]((nb,), (threads,), (x, out, sums, np.int64(m)))
        if nb > 1:
            self._scan(kind, suffix, sums, scanned, levels, depth + 1)
            self.k[f"scan_add_offsets_{suffix}"]((nb,), (256,), (out, scanned, np.int64(m)))


def build_cub() -> ctypes.CDLL:
    out = REPO_ROOT / "build" / "libscan_cub.so"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["nvcc", "-O3", "-arch=sm_75", "-Xcompiler", "-fPIC", "-shared", "-o", str(out),
           str(REPO_ROOT / "cuda" / "scan_cub.cu")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed: {(proc.stderr or proc.stdout)[-2000:]}")
    lib = ctypes.CDLL(str(out))
    for name in ("cub_inclusive_sum_f64", "cub_inclusive_sum_f32"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
    return lib


def scan_table(bench: Bench, sc: Scans, values: np.ndarray, label: str) -> dict:
    cp = sc.cp
    x = cp.asarray(values.astype(np.float64))
    outs = {k: cp.empty_like(x) for k in ("blelloch", "warpscan", "cub")}
    want = np.cumsum(values.astype(np.float64))
    kinds = ["blelloch", "warpscan"] + (["cub"] if sc.cub is not None else [])
    verified = {}
    for kind in kinds:
        outs[kind].fill(0)
        sc.scan(kind, x, outs[kind])
        got = cp.asnumpy(outs[kind])
        err = np.abs(got - want) / np.maximum(np.abs(want), 1e-300)
        verified[kind] = {"verified": bool(err.max() <= 1e-9), "max_rel_err": float(err.max()),
                          "final_rel_err": float(err[-1])}
    names = {"blelloch": "blelloch_shared", "warpscan": "warp_shuffle", "cub": "cub_device_scan"}
    res = bench.block(f"scan_{label}", {names[k]: (lambda k=k: event_time_ms(sc.stream, lambda: sc.scan(k, x, outs[k])))
                                       for k in kinds})
    n = int(values.size)
    impls = []
    notes = {"blelloch": "work-efficient up-sweep/down-sweep, 1024 elements per block, bank-conflict padding",
             "warpscan": "__shfl_up_sync warp scan, 256 threads x 4 elements, warp totals scanned by warp 0",
             "cub": "cub::DeviceScan::InclusiveSum, CUDA toolkit CUB, nvcc -O3 -arch=sm_75"}
    for kind in kinds:
        m = res[names[kind]]
        impls.append({"name": names[kind], "ms": m, "elements_per_s": n / (m["median_ms"] * 1e-3),
                      "gbs": 16 * n / (m["median_ms"] * 1e-3) / 1e9, "notes": notes[kind], **verified[kind]})
    if sc.cub is None:
        impls.append({"name": "cub_device_scan", "ms": None, "verified": False, "error": sc.cub_error})
    timed = [i for i in impls if i.get("ms")]
    winner = min(timed, key=lambda i: i["ms"]["median_ms"])["name"]
    del x, outs
    return {"n_elements": n, "dtype": "float64", "implementations": impls, "winner": winner,
            "bytes_per_element": 16, "counting": "8 bytes read + 8 bytes written per element"}


def run(bench: Bench, arch: dict, quick: bool = False) -> None:
    cp = cupy()
    t0 = time.perf_counter()
    meta = arch["meta"]
    n_products = len(meta["products"])
    n_total = int(meta["n_snapshots"])
    n = min(n_total, QUICK_N) if quick else n_total
    sc = Scans()

    # 1. scan implementations on a real series: top-of-book bid notional per snapshot, in ticks x size
    series = arch["bid_px"][0][:n].astype(np.float64) * arch["bid_sz"][0][:n].astype(np.float64)
    main = scan_table(bench, sc, series, "snapshots")
    large_n = QUICK_LARGE_N if quick else LARGE_N
    large = scan_table(bench, sc, np.resize(series, large_n), "large")
    log(f"scans n={n:,}: winner {main['winner']}; n={large_n:,}: winner {large['winner']}")
    scan_kind = {"cub_device_scan": "cub", "warp_shuffle": "warpscan", "blelloch_shared": "blelloch"}[large["winner"]]

    # 2. rolling VWAP over trades
    t_prod = arch["trade_product"].astype(np.int64)
    t_order, t_s, t_e = ref.product_segments(t_prod, n_products)
    t_px = arch["trade_px"][t_order].astype(np.float64)
    t_sz = arch["trade_sz"][t_order].astype(np.float64)
    t_ts = arch["trade_ts"][t_order].astype(np.int64)
    t_prod_sorted = t_prod[t_order].astype(np.uint8)
    pv = t_px * t_sz
    d_pv, d_sz = cp.asarray(pv), cp.asarray(t_sz)
    d_P, d_S = cp.empty_like(d_pv), cp.empty_like(d_sz)
    d_Pl, d_Sl = cp.empty_like(d_pv), cp.empty_like(d_sz)
    dds = DDScan()
    d_tprod, d_tts = cp.asarray(t_prod_sorted), cp.asarray(t_ts)
    d_ts_start, d_ts_end = cp.asarray(t_s), cp.asarray(t_e)
    n_tr = int(pv.size)
    d_vwap100 = cp.empty(n_tr, dtype=cp.float64)
    snap_ts = arch["snap_ts"][:n].astype(np.int64)
    snap_prod = arch["snap_product"][:n].astype(np.uint8)
    d_snap_ts, d_snap_prod = cp.asarray(snap_ts), cp.asarray(snap_prod)
    d_vwap_t = {w: cp.empty(n, dtype=cp.float64) for w in ref.TIME_WINDOWS_NS}
    roll = raw_module(["rolling.cu"])
    k_tr, k_time = roll.get_function("vwap_trade_window_dd"), roll.get_function("vwap_time_window_dd")
    k_book, k_mean = roll.get_function("book_features_sorted"), roll.get_function("window_mean")

    def vwap_chain() -> None:  # production design: double-double prefixes (see the precision study)
        dds.scan(d_pv, d_P, d_Pl)
        dds.scan(d_sz, d_S, d_Sl)
        k_tr((max(1, math.ceil(n_tr / BLOCK)),), (BLOCK,), (d_P, d_Pl, d_S, d_Sl, d_tprod, d_ts_start, np.int64(n_tr),
                                                          np.int32(ref.VWAP_WINDOW_TRADES), d_vwap100))
        for w in ref.TIME_WINDOWS_NS:
            k_time((max(1, math.ceil(n / BLOCK)),), (BLOCK,), (d_snap_ts, d_snap_prod, np.int64(n), d_tts, d_P, d_Pl,
                                                               d_S, d_Sl, d_ts_start, d_ts_end, np.int64(w), d_vwap_t[w]))

    vwap_chain()
    sc.stream.synchronize()
    vwap100 = cp.asnumpy(d_vwap100)
    want100 = ref.vwap_trades_numpy(pv, t_sz, t_prod_sorted, t_s)
    P_np, S_np = np.cumsum(pv), np.cumsum(t_sz)
    # Decimal check against metrics.rolling_vwap on sampled trades
    from exchange_engine.metrics import rolling_vwap

    rng = np.random.default_rng(20261021)
    sample = np.sort(rng.choice(n_tr, size=min(VWAP_SAMPLE, n_tr), replace=False)) if n_tr else np.zeros(0, int)
    tick = {i: Decimal(meta["tick_sizes"][p]) for i, p in enumerate(meta["products"])}
    dec_err = 0.0
    for i in sample.tolist():
        p = int(t_prod_sorted[i])
        lo = max(int(t_s[p]), i - ref.VWAP_WINDOW_TRADES + 1)
        trades = [{"price": Decimal(int(t_px[j])) * tick[p], "size": Decimal(str(float(arch["trade_sz"][t_order[j]])))}
                  for j in range(lo, i + 1)]
        want = rolling_vwap(trades, window=ref.VWAP_WINDOW_TRADES)
        got = Decimal(repr(float(vwap100[i]))) * tick[p]
        dec_err = max(dec_err, float(abs(got - want) / want))
    time_checks = {}
    snap_sample = np.sort(rng.choice(n, size=min(2000, n), replace=False)) if n else np.zeros(0, int)
    for w in ref.TIME_WINDOWS_NS:
        got = cp.asnumpy(d_vwap_t[w])
        want = ref.vwap_time_numpy(snap_ts, snap_prod, t_ts, P_np, S_np, t_s, t_e, w)
        nan_ok = bool(np.array_equal(np.isnan(got), np.isnan(want)))
        fin = np.isfinite(want)
        np_err = float(np.max(np.abs(got[fin] - want[fin]) / np.abs(want[fin]))) if fin.any() else 0.0
        exact = ref.vwap_time_exact(snap_sample, snap_ts, snap_prod, t_ts, pv, t_sz, t_s, t_e, w)
        g = got[snap_sample]
        ok = np.isfinite(exact)
        err = float(np.max(np.abs(g[ok] - exact[ok]) / np.abs(exact[ok]))) if ok.any() else 0.0
        np_vs_exact = float(np.max(np.abs(want[snap_sample][ok] - exact[ok]) / np.abs(exact[ok]))) if ok.any() else 0.0
        time_checks[f"{w // 1_000_000_000}s"] = {"nan_pattern_equal": nan_ok, "max_rel_err": err,
                                                 "reference": "math.fsum over each sampled window",
                                                 "sample": int(snap_sample.size),
                                                 "float64_cumsum_max_rel_err": np_vs_exact,
                                                 "gpu_vs_numpy_cumsum_max_rel_err": np_err,
                                                 "coverage_pct": float(fin.mean() * 100.0)}
    vwap_ms = bench.block("rolling_vwap", {"vwap_chain": lambda: event_time_ms(sc.stream, vwap_chain)})["vwap_chain"]
    rolling_vwap_section = {
        "n_trades": n_tr, "n_snapshots": n, "windows": ["100_trades", "1s", "10s"], "ms": vwap_ms,
        "scan": "double_double", "verified_vs_metrics_py": bool(dec_err <= 1e-9), "max_rel_err_100": dec_err,
        "decimal_sample": int(sample.size),
        "numpy_100_max_rel_err": float(np.nanmax(np.abs(vwap100 - want100) / np.abs(want100))) if n_tr else 0.0,
        "time_windows": time_checks,
        "verified": bool(dec_err <= 1e-9 and all(c["nan_pattern_equal"] and c["max_rel_err"] <= 1e-9
                                                 for c in time_checks.values())),
        "definition": "two inclusive double-double scans (ticks x size, size) over product-contiguous trades, "
                      "windowed difference; time windows checked against math.fsum on sampled snapshots",
    }
    log(f"rolling vwap: {vwap_ms['median_ms']:.3f} ms, decimal max rel err {dec_err:.2e}, {time_checks}")

    # 3. rolling 1 s mean spread and imbalance over the snapshot stream
    s_order, s_s, _s_e = ref.product_segments(snap_prod.astype(np.int64), n_products)
    stride = int(arch["bid_px"].shape[1])
    d_bpx = cp.asarray(np.ascontiguousarray(arch["bid_px"][0]))  # level 0 row, full archive width
    d_apx = cp.asarray(np.ascontiguousarray(arch["ask_px"][0]))
    d_bsz = cp.asarray(np.ascontiguousarray(arch["bid_sz"][:10]))
    d_asz = cp.asarray(np.ascontiguousarray(arch["ask_sz"][:10]))
    d_order = cp.asarray(s_order)
    ts_sorted = snap_ts[s_order]
    prod_sorted = snap_prod[s_order]
    d_ts_sorted, d_prod_sorted, d_s_start = cp.asarray(ts_sorted), cp.asarray(prod_sorted), cp.asarray(s_s)
    d_spread, d_imb = cp.empty(n, cp.float64), cp.empty(n, cp.float64)
    d_Psp, d_Pim = cp.empty(n, cp.float64), cp.empty(n, cp.float64)
    d_msp, d_mim = cp.empty(n, cp.float64), cp.empty(n, cp.float64)
    grid = (max(1, math.ceil(n / BLOCK)),)

    def book_chain() -> None:
        k_book(grid, (BLOCK,), (d_bpx, d_bsz, d_apx, d_asz, d_order, np.int64(n), np.int64(stride), d_spread, d_imb))
        sc.scan(scan_kind, d_spread, d_Psp)
        sc.scan(scan_kind, d_imb, d_Pim)
        k_mean(grid, (BLOCK,), (d_ts_sorted, d_prod_sorted, d_s_start, d_Psp, np.int64(n), np.int64(ref.BOOK_WINDOW_NS), d_msp))
        k_mean(grid, (BLOCK,), (d_ts_sorted, d_prod_sorted, d_s_start, d_Pim, np.int64(n), np.int64(ref.BOOK_WINDOW_NS), d_mim))

    book_chain()
    sc.stream.synchronize()
    sp_ref, im_ref = ref.book_series_numpy(arch["bid_px"], arch["bid_sz"], arch["ask_px"], arch["ask_sz"], s_order)
    msp_ref = ref.window_mean_numpy(ts_sorted, prod_sorted, s_s, sp_ref)
    mim_ref = ref.window_mean_numpy(ts_sorted, prod_sorted, s_s, im_ref)
    err_sp = float(np.max(np.abs(cp.asnumpy(d_msp) - msp_ref) / np.maximum(np.abs(msp_ref), 1e-12)))
    err_im = float(np.max(np.abs(cp.asnumpy(d_mim) - mim_ref)))
    book_ms = bench.block("rolling_book", {"book_chain": lambda: event_time_ms(sc.stream, book_chain)})["book_chain"]
    rolling_book = {"n_snapshots": n, "window_s": ref.BOOK_WINDOW_NS / 1e9, "ms": book_ms, "scan": scan_kind,
                    "max_rel_err_spread": err_sp, "max_abs_err_imbalance": err_im,
                    "verified": bool(err_sp <= 1e-9 and err_im <= 1e-6),
                    "snapshots_per_s": n / (book_ms["median_ms"] * 1e-3),
                    "definition": "mean over the same product's snapshots with time in (t - 1 s, t]"}
    log(f"rolling book: {book_ms['median_ms']:.3f} ms, spread err {err_sp:.2e}, imbalance err {err_im:.2e}")

    bench.section("scans", {**main, "large": large, "rolling_vwap": rolling_vwap_section, "rolling_book": rolling_book,
                            "cub_error": sc.cub_error, "quick": bool(quick), "wall_seconds": time.perf_counter() - t0,
                            "series": "top-of-book bid notional per snapshot (ticks x size), float64"})
    cp.get_default_memory_pool().free_all_blocks()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4: scans and rolling windows")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", required=True, choices=("synthetic", "real"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    arch = load_archive(Path(args.archive))
    run(Bench(Path(args.out), args.source, arch["meta"]), arch, quick=args.quick)


if __name__ == "__main__":
    main()
