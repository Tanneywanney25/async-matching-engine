"""Phase 5 benchmark: the execution-cost surface on the GPU.

    python -m bench.bench_slippage --archive data/synth_small --out results/x --source synthetic [--quick]

For every snapshot, both sides and the 64 log-spaced notional sizes in
:data:`bench.ref_slippage.SIZES_USD`, ``cuda/slippage.cu`` computes slippage
against mid, the average fill offset, the fill fraction and the number of
levels used. Two kernels are benchmarked interleaved with CUDA events around
the launches of a full pass over the archive:

* ``naive``: one thread per (snapshot, side, size) walking levels until
  filled (data-dependent loop length, divergent);
* ``warp``: one warp per snapshot, cumulative notional and quantity built
  with warp shuffles, each lane resolving two sizes by binary search.

The archive stays resident on the device; outputs are produced per chunk of
``CHUNK_SNAPSHOTS`` snapshots into reused buffers (13 bytes per simulated
order) so device memory stays bounded. After timing, one more pass gathers the
outputs for verification against the Decimal walker on a seeded sample of
snapshots, against the NumPy reference over the whole archive, and builds the
per-minute median surface for the dashboard. Section name: ``slippage``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from bench import ref_slippage as ref
from bench.common import Bench, cupy, event_time_ms, load_archive, raw_module

SECTION = "slippage"
LEVEL_ARRAYS = ("bid_px", "bid_sz", "ask_px", "ask_sz")
ORDERS_PER_SNAPSHOT = 2 * ref.N_SIZES
OUTPUT_BYTES_PER_ORDER = 4 + 4 + 4 + 1

CHUNK_SNAPSHOTS = 200_000
SAMPLE_SNAPSHOTS = 1000
SAMPLE_SEED = 20261021

QUICK_SNAPSHOTS = 20_000
QUICK_CHUNK = 8_192
QUICK_ITERS = 5
QUICK_SAMPLE = 200

# Launch configuration (must match the constants in cuda/slippage.cu).
NAIVE_BLOCK = 256
WARPS_PER_BLOCK = 8
WARP_BLOCK = 32 * WARPS_PER_BLOCK
WARP_SHARED_BYTES = WARPS_PER_BLOCK * 64 * (8 + 8 + 4)


def plan_chunks(n: int, chunk: int) -> List[Tuple[int, int]]:
    """``[(i0, count), ...]`` covering ``[0, n)`` in pieces of at most ``chunk``."""
    chunk = max(1, int(chunk))
    return [(i0, min(chunk, n - i0)) for i0 in range(0, int(n), chunk)]


class SlippageGPU:
    """The archive (or its first ``n`` snapshots) resident on the device plus both kernels."""

    def __init__(self, arch: dict, n: Optional[int] = None, chunk: int = CHUNK_SNAPSHOTS) -> None:
        cp = cupy()
        self.cp = cp
        meta = arch["meta"]
        self.products = list(meta["products"])
        self.tick_sizes = dict(meta["tick_sizes"])
        n_total = int(meta["n_snapshots"])
        self.n = n_total if n is None else max(0, min(int(n), n_total))
        self.chunk = max(1, min(int(chunk), self.n)) if self.n else 1
        self.chunks = plan_chunks(self.n, self.chunk)
        self.targets = ref.targets_ticks(self.products, self.tick_sizes)
        self.host = {name: np.ascontiguousarray(arch[name][:, : self.n]) for name in LEVEL_ARRAYS}
        self.product = np.ascontiguousarray(arch["snap_product"][: self.n])
        self.snap_ts = np.ascontiguousarray(arch["snap_ts"][: self.n])
        self.levels = int(self.host["bid_px"].shape[0])
        self.dev = {name: cp.asarray(a) for name, a in self.host.items()}
        self.d_product = cp.asarray(self.product)
        self.d_targets = cp.asarray(self.targets)
        max_chunk = max((c for _, c in self.chunks), default=1)
        m = max_chunk * ORDERS_PER_SNAPSHOT
        self.out = {
            "slip_bps": cp.empty(m, dtype=cp.float32),
            "fill_frac": cp.empty(m, dtype=cp.float32),
            "avg_offset_ticks": cp.empty(m, dtype=cp.float32),
            "levels_used": cp.empty(m, dtype=cp.uint8),
        }
        self.module = raw_module(["slippage.cu"])
        self.kernels = {
            "naive": self.module.get_function("slippage_naive"),
            "warp": self.module.get_function("slippage_warp"),
        }
        self.stream = cp.cuda.get_current_stream()

    # -- launches ----------------------------------------------------------- #

    def _args(self, i0: int, count: int) -> tuple:
        return (self.dev["bid_px"], self.dev["bid_sz"], self.dev["ask_px"], self.dev["ask_sz"],
                self.d_product, self.d_targets, np.int64(self.n), np.int64(i0), np.int32(count),
                self.out["slip_bps"], self.out["fill_frac"], self.out["avg_offset_ticks"], self.out["levels_used"])

    def grid(self, kernel: str, count: int) -> int:
        if kernel == "naive":
            return math.ceil(count * ORDERS_PER_SNAPSHOT / NAIVE_BLOCK)
        return math.ceil(count / WARPS_PER_BLOCK)

    def launch(self, kernel: str, i0: int, count: int) -> None:
        block = NAIVE_BLOCK if kernel == "naive" else WARP_BLOCK
        self.kernels[kernel]((self.grid(kernel, count),), (block,), self._args(i0, count))

    def full_pass_ms(self, kernel: str) -> float:
        """GPU time (CUDA events) of one full pass of ``kernel`` over every chunk."""
        return event_time_ms(self.stream, lambda: [self.launch(kernel, i0, c) for i0, c in self.chunks])

    def chunk_outputs(self, kernel: str, i0: int, count: int) -> Dict[str, np.ndarray]:
        """Run ``kernel`` on one chunk and return its outputs on the host as ``(count, 2, 64)``."""
        self.launch(kernel, i0, count)
        self.stream.synchronize()
        m = count * ORDERS_PER_SNAPSHOT
        return {name: self.cp.asnumpy(buf[:m]).reshape(count, 2, ref.N_SIZES) for name, buf in self.out.items()}

    def all_outputs(self, kernel: str) -> Dict[str, np.ndarray]:
        """Every chunk's outputs concatenated (host memory: 13 bytes per order)."""
        parts = [self.chunk_outputs(kernel, i0, c) for i0, c in self.chunks]
        return {name: np.concatenate([p[name] for p in parts], axis=0) for name in self.out}

    def device_bytes(self) -> int:
        total = sum(int(a.nbytes) for a in self.dev.values())
        total += int(self.d_product.nbytes) + int(self.d_targets.nbytes)
        total += sum(int(b.nbytes) for b in self.out.values())
        return total


def _agreement(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> dict:
    """How far apart two kernels' outputs are (NaN in both counts as equal)."""
    out = {"levels_used_mismatch": int(np.count_nonzero(a["levels_used"] != b["levels_used"]))}
    for name in ("slip_bps", "fill_frac", "avg_offset_ticks"):
        x = a[name].astype(np.float64)
        y = b[name].astype(np.float64)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.abs(x - y)
        diff[both_nan] = 0.0
        out[f"max_abs_{name}_diff"] = float(np.nanmax(diff)) if diff.size else 0.0
        out[f"nan_mismatch_{name}"] = int(np.count_nonzero(np.isnan(x) != np.isnan(y)))
    return out


def _merge_agreement(parts: List[dict]) -> dict:
    out = dict(parts[0])
    for p in parts[1:]:
        for key, val in p.items():
            out[key] = max(out[key], val) if key.startswith("max_") else out[key] + val
    return out


def collect(gpu: SlippageGPU, sample_idx: np.ndarray, numpy_check: bool = True) -> dict:
    """One production pass per kernel: surface input, sample rows, NumPy and cross-kernel checks."""
    n_sizes = ref.N_SIZES
    slip_all = np.empty((gpu.n, 2, n_sizes), dtype=np.float32)
    sample: Dict[str, Dict[str, np.ndarray]] = {}
    for kernel in ("naive", "warp"):
        sample[kernel] = {
            "slip_bps": np.empty((sample_idx.size, 2, n_sizes), np.float32),
            "fill_frac": np.empty((sample_idx.size, 2, n_sizes), np.float32),
            "avg_offset_ticks": np.empty((sample_idx.size, 2, n_sizes), np.float32),
            "levels_used": np.empty((sample_idx.size, 2, n_sizes), np.uint8),
        }
    numpy_parts: Dict[str, List[dict]] = {"naive": [], "warp": []}
    agree_parts: List[dict] = []
    numpy_seconds = 0.0
    for i0, count in gpu.chunks:
        outs = {kernel: gpu.chunk_outputs(kernel, i0, count) for kernel in ("warp", "naive")}
        slip_all[i0:i0 + count] = outs["warp"]["slip_bps"]
        in_chunk = (sample_idx >= i0) & (sample_idx < i0 + count)
        rows = np.flatnonzero(in_chunk)
        local = sample_idx[in_chunk] - i0
        for kernel in ("naive", "warp"):
            for name, arr in outs[kernel].items():
                sample[kernel][name][rows] = arr[local]
        agree_parts.append(_agreement(outs["naive"], outs["warp"]))
        if numpy_check:
            sl = slice(i0, i0 + count)
            t0 = time.perf_counter()
            np_ref = ref.numpy_slippage(gpu.host["bid_px"][:, sl], gpu.host["bid_sz"][:, sl],
                                        gpu.host["ask_px"][:, sl], gpu.host["ask_sz"][:, sl],
                                        gpu.product[sl], gpu.targets)
            numpy_seconds += time.perf_counter() - t0
            for kernel in ("naive", "warp"):
                numpy_parts[kernel].append(ref.compare_outputs(
                    outs[kernel], np_ref, gpu.host["bid_px"][:, sl], gpu.host["ask_px"][:, sl]))
        del outs
    return {
        "slip_all": slip_all,
        "sample": sample,
        "numpy": {k: ref.merge_comparisons(v) for k, v in numpy_parts.items()} if numpy_check else {},
        "numpy_seconds": numpy_seconds,
        "agreement": _merge_agreement(agree_parts),
    }


def verify_decimal(gpu: SlippageGPU, sample_idx: np.ndarray, sample: Dict[str, Dict[str, np.ndarray]]) -> dict:
    """Both kernels' sample rows against the Decimal walker."""
    ticks = ref.ticks_by_product(gpu.products, gpu.tick_sizes)
    t0 = time.perf_counter()
    dec = ref.decimal_sample(gpu.host["bid_px"], gpu.host["bid_sz"], gpu.host["ask_px"], gpu.host["ask_sz"],
                             gpu.product, sample_idx, ticks)
    seconds = time.perf_counter() - t0
    bid_cols = gpu.host["bid_px"][:, sample_idx]
    ask_cols = gpu.host["ask_px"][:, sample_idx]
    out = {kernel: ref.compare_outputs(sample[kernel], dec, bid_cols, ask_cols) for kernel in sample}
    for kernel in out:
        out[kernel]["sample"] = int(sample_idx.size)
        out[kernel]["seed"] = SAMPLE_SEED
        out[kernel]["reference"] = "Decimal walker, bench/ref_slippage.py (ticks * Decimal('0.01'), Decimal(str(float(f32))))"
    out["decimal_seconds"] = seconds
    return out


def run(bench: Bench, arch: dict, quick: bool = False) -> None:
    """Benchmark both kernels, verify, export the surface and write the ``slippage`` section."""
    t_start = time.perf_counter()
    n = QUICK_SNAPSHOTS if quick else None
    gpu = SlippageGPU(arch, n=n, chunk=QUICK_CHUNK if quick else CHUNK_SNAPSHOTS)
    if gpu.n == 0:
        raise ValueError("archive has no snapshots")

    summaries = bench.block("slippage_full_pass",
                            {"naive": lambda: gpu.full_pass_ms("naive"), "warp": lambda: gpu.full_pass_ms("warp")},
                            n_iters=QUICK_ITERS if quick else None)
    orders = gpu.n * ORDERS_PER_SNAPSHOT
    naive_ms = summaries["naive"]["median_ms"]
    warp_ms = summaries["warp"]["median_ms"]

    rng = np.random.default_rng(SAMPLE_SEED)
    sample_n = min(QUICK_SAMPLE if quick else SAMPLE_SNAPSHOTS, gpu.n)
    sample_idx = np.sort(rng.choice(gpu.n, size=sample_n, replace=False)).astype(np.int64)

    got = collect(gpu, sample_idx, numpy_check=True)
    decimal = verify_decimal(gpu, sample_idx, got["sample"])
    surface, text = ref.build_surface(got["slip_all"], gpu.snap_ts, gpu.product, gpu.products)
    surface_path = bench.raw_dir / "slippage_surface.json"
    surface_path.write_text(text, encoding="utf-8")
    bench.save_raw("slippage_verification_sample", {
        "seed": SAMPLE_SEED, "sample_idx": sample_idx.tolist(),
        "warp": decimal["warp"], "naive": decimal["naive"], "decimal_seconds": decimal["decimal_seconds"],
    })

    def kernel_entry(name: str, median_ms: float) -> dict:
        entry = {"ms": summaries[name], "orders_per_s": orders / (median_ms * 1e-3) if median_ms > 0 else 0.0,
                 "snapshots_per_s": gpu.n / (median_ms * 1e-3) if median_ms > 0 else 0.0,
                 "launches_per_pass": len(gpu.chunks)}
        if name == "naive":
            entry.update({"block_size": NAIVE_BLOCK, "threads_per_order": 1,
                          "grid_per_full_chunk": gpu.grid("naive", gpu.chunk),
                          "design": "one thread per (snapshot, side, size), data-dependent walk"})
        else:
            entry.update({"block_size": WARP_BLOCK, "warps_per_block": WARPS_PER_BLOCK,
                          "shared_bytes_per_block": WARP_SHARED_BYTES,
                          "grid_per_full_chunk": gpu.grid("warp", gpu.chunk),
                          "design": "one warp per snapshot, shuffle scans, binary search per lane"})
        return entry

    section = {
        "n_snapshots": int(gpu.n),
        "n_snapshots_archive": int(arch["meta"]["n_snapshots"]),
        "n_sizes": ref.N_SIZES,
        "sizes_usd": [float(x) for x in ref.SIZES_USD],
        "sides": 2,
        "levels": gpu.levels,
        "orders": int(orders),
        "chunk_snapshots": int(gpu.chunk),
        "chunks": len(gpu.chunks),
        "quick": bool(quick),
        "timing": "CUDA events around all chunk launches of one full pass (per-kernel GPU time, no transfers)",
        "naive": kernel_entry("naive", naive_ms),
        "warp": kernel_entry("warp", warp_ms),
        "speedup_warp_vs_naive": naive_ms / warp_ms if warp_ms > 0 else 0.0,
        "divergence_penalty_pct": (naive_ms - warp_ms) / naive_ms * 100.0 if naive_ms > 0 else 0.0,
        "verification": decimal["warp"],
        "verification_naive": decimal["naive"],
        "numpy_check": {"n_snapshots": int(gpu.n), "warp": got["numpy"].get("warp", {}),
                        "naive": got["numpy"].get("naive", {}), "numpy_seconds": got["numpy_seconds"]},
        "naive_vs_warp": got["agreement"],
        "surface_file": "raw/slippage_surface.json",
        "surface_bytes": len(text.encode("utf-8")),
        "surface_bucket_minutes": int(surface["bucket_minutes"]),
        "surface_buckets": {p: len(v["buy"]["minutes"]) for p, v in surface["products"].items()},
        "output_bytes_per_order": OUTPUT_BYTES_PER_ORDER,
        "output_bytes_per_chunk": int(gpu.chunk * ORDERS_PER_SNAPSHOT * OUTPUT_BYTES_PER_ORDER),
        "device_bytes_resident": gpu.device_bytes(),
        "arithmetic": "float64 cumulative notional (px_ticks * sz) and float64 cumulative quantity on the device",
        "tick_sizes": gpu.tick_sizes,
        "wall_seconds": time.perf_counter() - t_start,
    }
    bench.section(SECTION, section)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 5: execution-cost surface benchmark (GPU)")
    parser.add_argument("--archive", required=True, help="archive directory (archive.json + arrays)")
    parser.add_argument("--out", required=True, help="results/<run_id> directory")
    parser.add_argument("--source", required=True, choices=("synthetic", "real"))
    parser.add_argument("--quick", action="store_true", help="few iterations on a subset (smoke run)")
    args = parser.parse_args(argv)
    arch = load_archive(Path(args.archive))
    bench = Bench(Path(args.out), args.source, arch["meta"])
    run(bench, arch, quick=args.quick)
    s = bench.metrics["sections"][SECTION]
    print(f"slippage: {s['n_snapshots']:,} snapshots x {s['orders'] // max(1, s['n_snapshots'])} orders -> "
          f"naive {s['naive']['ms']['median_ms']:.2f} ms ({s['naive']['orders_per_s']:.3g} orders/s), "
          f"warp {s['warp']['ms']['median_ms']:.2f} ms ({s['warp']['orders_per_s']:.3g} orders/s), "
          f"speedup {s['speedup_warp_vs_naive']:.2f}x, divergence penalty {s['divergence_penalty_pct']:.1f}%, "
          f"verification passed={s['verification']['passed']} exact_ticks={s['verification']['exact_ticks']}")


if __name__ == "__main__":
    main()
