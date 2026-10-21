"""Phase 6 benchmark: order flow imbalance stencil on the GPU plus the regression study.

    python -m bench.bench_ofi --archive data/synth_small --out results/x --source synthetic [--quick]

``cuda/ofi.cu`` computes the K = 10 level OFI of Cont, Kukanov and Stoikov
(2014) between consecutive snapshots of the same product as a stencil over a
stable product-sorted index array built once on the host. ``kernel_ms`` is
CUDA events around that launch (at least 30 interleaved iterations, beside a
host NumPy timing of the same computation). The regression part (horizons of
1 s and 10 s, time-ordered 70/30 split, OLS with intercept, out-of-sample
R^2) runs in NumPy on the kernel's output. Section name: ``ofi``; scatter
sample in ``raw/ofi_scatter.json``. No trading claims anywhere.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from bench import ref_ofi as ref
from bench.common import Bench, cupy, effective_bandwidth, event_time_ms, load_archive, raw_module

SECTION = "ofi"
LEVEL_ARRAYS = ("bid_px", "bid_sz", "ask_px", "ask_sz")
OFI_BLOCK = 256  # matches OFI_BLOCK in cuda/ofi.cu
QUICK_SNAPSHOTS = 20_000
QUICK_ITERS = 5
MISSING_LEVEL_CONVENTION = ("missing level (px = 0, sz = 0) takes the worst price of its side before the "
                            "comparison: 0 for a bid, INT_MAX for an ask")


class OfiGPU:
    """The first ``levels`` rows of the archive (or its first ``n`` snapshots) on the device."""

    def __init__(self, arch: dict, n: Optional[int] = None, levels: int = ref.OFI_LEVELS) -> None:
        cp = cupy()
        self.cp = cp
        meta = arch["meta"]
        self.products = list(meta["products"])
        n_total = int(meta["n_snapshots"])
        self.n = n_total if n is None else max(0, min(int(n), n_total))
        self.levels = min(int(levels), int(arch["bid_px"].shape[0]))
        self.host = {name: np.ascontiguousarray(arch[name][: self.levels, : self.n]) for name in LEVEL_ARRAYS}
        self.product = np.ascontiguousarray(arch["snap_product"][: self.n])
        self.snap_ts = np.ascontiguousarray(arch["snap_ts"][: self.n])
        self.order, self.ranges = ref.product_order(self.product)
        self.dev = {name: cp.asarray(a) for name, a in self.host.items()}
        self.d_product = cp.asarray(self.product)
        self.d_order = cp.asarray(self.order.astype(np.int32))
        self.d_out = cp.empty(max(1, self.n), dtype=cp.float32)
        self.module = raw_module(["ofi.cu"])
        self.kernel = self.module.get_function("ofi_stencil")
        self.grid = max(1, math.ceil(self.n / OFI_BLOCK))
        self.stream = cp.cuda.get_current_stream()

    def launch(self) -> None:
        self.kernel((self.grid,), (OFI_BLOCK,),
                    (self.dev["bid_px"], self.dev["bid_sz"], self.dev["ask_px"], self.dev["ask_sz"],
                     self.d_product, self.d_order, np.int64(self.n), np.int32(self.levels), self.d_out))

    def kernel_ms(self) -> float:
        """GPU time of one stencil launch (CUDA events)."""
        return event_time_ms(self.stream, self.launch)

    def result(self) -> np.ndarray:
        """OFI in sorted-position order (``result()[pos]`` belongs to ``order[pos]``)."""
        self.launch()
        self.stream.synchronize()
        return self.cp.asnumpy(self.d_out[: self.n])

    def numpy_ms(self) -> float:
        """Wall time of the NumPy reference on the same host arrays (for an honest host figure)."""
        t0 = time.perf_counter()
        ref.ofi_numpy(self.host["bid_px"], self.host["bid_sz"], self.host["ask_px"], self.host["ask_sz"],
                      self.product, self.levels, order=self.order)
        return (time.perf_counter() - t0) * 1e3

    def min_io_bytes(self) -> int:
        """Minimum traffic: every level of every snapshot read once (4 arrays) plus the output."""
        return int(self.n) * self.levels * (4 + 4 + 4 + 4) + int(self.n) * (4 + 1) + int(self.n) * 4


def run(bench: Bench, arch: dict, quick: bool = False) -> None:
    """Time the stencil, verify it against NumPy, run the regression study, write the ``ofi`` section."""
    t_start = time.perf_counter()
    gpu = OfiGPU(arch, n=QUICK_SNAPSHOTS if quick else None)
    if gpu.n == 0:
        raise ValueError("archive has no snapshots")
    summaries = bench.block("ofi_stencil", {"gpu_stencil": gpu.kernel_ms, "numpy_host": gpu.numpy_ms},
                            n_iters=QUICK_ITERS if quick else None)

    ofi_gpu = gpu.result().astype(np.float64)
    ofi_ref, _ = ref.ofi_numpy(gpu.host["bid_px"], gpu.host["bid_sz"], gpu.host["ask_px"], gpu.host["ask_sz"],
                               gpu.product, gpu.levels, order=gpu.order)
    err = np.abs(ofi_gpu - ofi_ref)
    scale = max(1.0, float(np.max(np.abs(ofi_ref))) if ofi_ref.size else 1.0)
    max_abs_err = float(err.max()) if err.size else 0.0
    verified = bool(max_abs_err <= 1e-5 * scale)

    per_product, scatter = ref.ofi_study(ofi_gpu, gpu.order, gpu.product, gpu.snap_ts,
                                         gpu.host["bid_px"][0], gpu.host["ask_px"][0], gpu.products)
    scatter_file = bench.save_raw("ofi_scatter", scatter)
    bench.save_raw("ofi_series_summary", {
        "n_snapshots": int(gpu.n),
        "per_product_n": {name: int(gpu.ranges.get(pi, (0, 0))[1] - gpu.ranges.get(pi, (0, 0))[0])
                          for pi, name in enumerate(gpu.products)},
        "ofi_mean": float(ofi_gpu.mean()), "ofi_std": float(ofi_gpu.std()),
        "ofi_abs_max": float(np.abs(ofi_gpu).max()),
    })
    kernel_ms = summaries["gpu_stencil"]["median_ms"]
    section = {
        "levels": int(gpu.levels),
        "horizons_s": [int(h) for h in ref.HORIZONS_S],
        "split": "time-ordered 70/30",
        "per_product": per_product,
        "kernel_ms": summaries["gpu_stencil"],
        "numpy_host_ms": summaries["numpy_host"],
        "kernel_vs_numpy_host": summaries["numpy_host"]["median_ms"] / kernel_ms if kernel_ms > 0 else 0.0,
        "scatter_file": scatter_file,
        "scatter_points_max": ref.SCATTER_POINTS,
        "n_snapshots": int(gpu.n),
        "n_snapshots_archive": int(arch["meta"]["n_snapshots"]),
        "quick": bool(quick),
        "block_size": OFI_BLOCK,
        "grid": int(gpu.grid),
        "bytes_min_io": gpu.min_io_bytes(),
        "effective_bw_gbs": effective_bandwidth(gpu.min_io_bytes(), kernel_ms),
        "snapshots_per_s": gpu.n / (kernel_ms * 1e-3) if kernel_ms > 0 else 0.0,
        "verified_vs_numpy": verified,
        "max_abs_err_vs_numpy": max_abs_err,
        "missing_level_convention": MISSING_LEVEL_CONVENTION,
        "feature": "x_t = OFI summed over (t - h, t]; y_t = mid(t + h) - mid(t) in ticks, mid at the last snapshot with ts <= t + h",
        "statement": "R^2 measures the fit of future mid change on OFI only; no trading or profit claim is made",
        "wall_seconds": time.perf_counter() - t_start,
    }
    bench.section(SECTION, section)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 6: order flow imbalance stencil and regression study")
    parser.add_argument("--archive", required=True, help="archive directory (archive.json + arrays)")
    parser.add_argument("--out", required=True, help="results/<run_id> directory")
    parser.add_argument("--source", required=True, choices=("synthetic", "real"))
    parser.add_argument("--quick", action="store_true", help="few iterations on a subset (smoke run)")
    args = parser.parse_args(argv)
    arch = load_archive(Path(args.archive))
    bench = Bench(Path(args.out), args.source, arch["meta"])
    run(bench, arch, quick=args.quick)
    s = bench.metrics["sections"][SECTION]
    parts = []
    for name, horizons in s["per_product"].items():
        for key, fit in horizons.items():
            parts.append(f"{name} {key}: R2 oos {fit['r2_oos']:.4f} (in {fit['r2_in']:.4f}, n_test {fit['n_test']})")
    print(f"ofi: {s['n_snapshots']:,} snapshots, K={s['levels']}, stencil {s['kernel_ms']['median_ms']:.3f} ms "
          f"(numpy host {s['numpy_host_ms']['median_ms']:.1f} ms), verified={s['verified_vs_numpy']}; " + "; ".join(parts))


if __name__ == "__main__":
    main()
