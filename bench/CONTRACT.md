# Results contract

Everything that reaches a document or the dashboard is read from
`results/<run_id>/metrics.json` by `scripts/render_docs.py`. This file fixes the
shape of that JSON so the kernels, the renderer, the dashboard and the checks
can be built in parallel.

## Run directory

```
results/<run_id>/
  manifest.json   written by the remote bootstrap (remote/base.py): run_id, job, git_sha,
                  source ("real" | "synthetic"), archive_hash, gpu_name, driver, cuda_runtime,
                  nvcc, nsys, cpu_model, cpu_cores, clocks_at_start, clocks_at_end, status,
                  wall_seconds, kaggle_kernel
  env.json        full environment dump from the same bootstrap
  metrics.json    the measurements (shape below)
  kernel.log      remote stdout/stderr
  raw/            per-benchmark samples (`<block>__<variant>.json` with ms arrays and the
                  nvidia-smi state before/after), nsys reports, CSV summaries, SVG, surfaces
```

`metrics.json` top level:

```json
{
  "source": "real",
  "archive": {"source", "n_snapshots", "n_trades", "content_hash", "capture_seconds",
              "per_product_snapshots", "levels", "bytes", "gap_count"},
  "gpu": {"name", "sm_count", "memory_gb", "compute_capability", "theoretical_bw_gbs"},
  "sections": { ... one key per section below ... }
}
```

The renderer refuses any `metrics.json` whose `source` is not `"real"` or whose manifest
`gpu_name` does not contain `T4`. It merges sections across runs, latest run winning per
section, and records provenance `{run_id, json_path}` for every number it prints.

## Measurement dict `M`

Produced by `bench.common.summarize` and `Bench.block`:

```json
{"median_ms", "iqr_ms", "q1_ms", "q3_ms", "min_ms", "max_ms", "mean_ms", "n", "warmup_discarded",
 "samples_file", "sm_clock_before_mhz", "sm_clock_after_mhz", "clock_drop_pct", "throttle_flag",
 "temperature_before_c", "temperature_after_c", "power_before_w", "power_after_w",
 "mem_clock_before_mhz", "mem_clock_after_mhz"}
```

`n >= 30`, warmup discarded, variants interleaved (`Bench.block` does all of this).

## Sections

### `bandwidth` (Phase 3)
```json
{"bytes": 1073741824, "copy_kernel": M, "cudamemcpy_d2d": M,
 "copy_kernel_gbs": 0.0, "cudamemcpy_gbs": 0.0, "ceiling_gbs": 0.0,
 "theoretical_gbs": 0.0, "theoretical_spec_gbs": 320.0}
```
GB/s count read + write bytes. `ceiling_gbs = max(copy_kernel_gbs, cudamemcpy_gbs)`;
`theoretical_gbs` from `cudaGetDeviceProperties` (2 * memClockRate * busWidth / 8).

### `ladder` (Phase 3)
```json
{"n_snapshots": 0, "levels": 50, "bytes_min_io": 0,
 "rungs": [
   {"name": "v0_aos", "label": "v0 AoS, one kernel per metric", "kernel_ms": M, "e2e_ms": M,
    "effective_bw_gbs": 0.0, "pct_of_ceiling": 0.0, "pct_of_theoretical": 0.0,
    "block_size": 256, "grid": 0, "launches_per_pass": 4, "verified": true,
    "max_abs_err": {"spread_ticks": 0, "depth_bid": 0.0, "imbalance": 0.0, "microprice": 0.0},
    "arithmetic_intensity_flop_per_byte": 0.0}
   , ... "v1_soa", "v2_soa_fused", "v3_soa_fused_vec4", "v4_soa_fused_vec4_sweep"],
 "sweep": {"rung": "v4_soa_fused_vec4_sweep",
           "table": [{"block_size": 64, "grid": 0, "grid_stride": true, "median_ms": 0.0, "iqr_ms": 0.0}, ...],
           "best": {"block_size": 0, "grid": 0, "median_ms": 0.0}},
 "speedups": {"best_rung": "v4_soa_fused_vec4_sweep",
              "kernel_vs_cpp_single": 0.0, "e2e_vs_numpy": 0.0,
              "kernel_vs_cpp_omp": 0.0, "kernel_vs_cupy": 0.0, "e2e_vs_cupy": 0.0},
 "outputs": ["spread_ticks", "spread_usd", "depth_bid", "depth_ask", "imbalance10", "microprice"]}
```
`kernel_ms` is CUDA events around the launch(es) only; `e2e_ms` is wall clock including the
host-to-device copy of the inputs and the device-to-host copy of the outputs.
`bytes_min_io` = bytes the fused computation must read (all four level-major arrays) plus
the output bytes. `kernel_vs_cpp_single` = cpp_single median / best rung kernel median
(bullet 1's `{A}`); `e2e_vs_numpy` = numpy median / best rung e2e median (`{B}`).

### `baselines` (Phase 3, same job, same archive)
```json
{"cpp_single": {"ms": M, "compiler": "g++ -O3 -march=native", "threads": 1, "verified": true},
 "cpp_omp": {"ms": M, "compiler": "g++ -O3 -march=native -fopenmp", "threads": 0, "verified": true},
 "numpy": {"ms": M, "numpy_version": "", "verified": true},
 "cupy": {"kernel_ms": M, "e2e_ms": M, "cupy_version": "", "n_elementwise_ops": 0, "verified": true}}
```

### `scans` (Phase 4)
```json
{"n_elements": 0, "dtype": "float64",
 "implementations": [{"name": "blelloch_shared", "ms": M, "elements_per_s": 0.0, "gbs": 0.0, "verified": true,
                      "notes": "work-efficient, bank-conflict padding"},
                     {"name": "warp_shuffle", ...}, {"name": "cub_device_scan", ...},
                     {"name": "decoupled_lookback", ..., "stretch": true}],
 "winner": "cub_device_scan",
 "rolling_vwap": {"n_trades": 0, "n_snapshots": 0, "windows": ["100_trades", "1s", "10s"],
                  "ms": M, "verified_vs_metrics_py": true, "max_rel_err_100": 0.0},
 "rolling_book": {"n_snapshots": 0, "window_s": 1.0, "ms": M, "verified": true}}
```

### `precision` (Phase 4)
```json
{"n": 0, "window": 100,
 "variants": [{"name": "float32_naive", "max_rel_err": 0.0, "median_rel_err": 0.0, "ms": M},
              {"name": "float64", ...}, {"name": "kahan_float32", ...}, {"name": "block_rebased_float32", ...}],
 "exact_reference": "python int over tick and 1e-8 size base units",
 "production_choice": "float64", "cost_vs_float32_naive_pct": 0.0, "rationale": ""}
```

### `layout` (Phase 4, Nsight Systems)
```json
{"aos_total_kernel_ms": 0.0, "soa_total_kernel_ms": 0.0, "reduction_pct": 0.0,
 "kernels": [{"name": "", "aos_ms": 0.0, "soa_ms": 0.0, "aos_calls": 0, "soa_calls": 0}],
 "nsys_version": "", "reports": {"aos": "raw/profiles/aos.nsys-rep", "soa": "raw/profiles/soa.nsys-rep",
                                  "aos_csv": "raw/profiles/aos_kern_sum.csv", "soa_csv": "raw/profiles/soa_kern_sum.csv"},
 "ncu": {"attempted": true, "ok": false, "message": ""}}
```
`reduction_pct` is bullet 2's `{C}`: `(aos - soa) / aos * 100` over total kernel time parsed from
the `cuda_gpu_kern_sum` CSVs.

### `slippage` (Phase 5)
```json
{"n_snapshots": 0, "n_sizes": 64, "sizes_usd": [1000.0, ..., 10000000.0], "sides": 2, "orders": 0,
 "naive": {"ms": M, "orders_per_s": 0.0}, "warp": {"ms": M, "orders_per_s": 0.0},
 "speedup_warp_vs_naive": 0.0, "divergence_penalty_pct": 0.0,
 "verification": {"sample": 1000, "exact_ticks": true, "max_last_level_mismatch": 0,
                  "max_avg_price_tick_err": 0.0, "max_fill_frac_err": 0.0},
 "surface_file": "raw/slippage_surface.json"}
```
`surface_file` holds per product and side a per-minute median of slippage (bps) for each size:
`{"sizes_usd": [...], "products": {"BTC-USD": {"buy": {"minutes": [...], "bps": [[...per size...], ...]}, "sell": ...}}}`
kept small enough for the dashboard (under 1 MB).

### `ofi` (Phase 6)
```json
{"levels": 10, "horizons_s": [1, 10], "split": "time-ordered 70/30",
 "per_product": {"BTC-USD": {"h1s": {"r2_oos": 0.0, "r2_in": 0.0, "beta": 0.0, "alpha": 0.0, "n_train": 0, "n_test": 0},
                             "h10s": {...}}, "ETH-USD": {...}},
 "kernel_ms": M, "scatter_file": "raw/ofi_scatter.json"}
```
`scatter_file`: `{"BTC-USD": {"h1s": {"x": [...], "y": [...]}}, ...}` with at most 2000 points each.

### `pipeline` (Phase 7)
```json
{"chunks": 0, "chunk_bytes": 0, "streams": 3, "pinned": true,
 "overlapped_ms": M, "serial_ms": M, "overlap_saved_pct": 0.0,
 "breakdown": {"h2d_ms": 0.0, "kernel_ms": 0.0, "d2h_ms": 0.0, "sum_ms": 0.0, "overlapped_wall_ms": 0.0},
 "h2d_gbs": 0.0, "d2h_gbs": 0.0,
 "timeline_svg": "raw/pipeline_timeline.svg", "nsys_report": "raw/pipeline.nsys-rep", "nsys_sqlite": "raw/pipeline.sqlite",
 "transfer_share_pct": 0.0, "consequence": "",
 "int16_encoding": {"attempted": true, "ok": true, "e2e_ms": M, "bytes_saved_pct": 0.0, "overflow_fallbacks": 0}}
```

### `engine` (Phase 8, run on the job's host CPU and locally)
```json
{"orders": 100000, "cancels": 10000, "python": "", "cpu_model": "",
 "add_order": {"mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0},
 "cancel_order": {"mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0},
 "raw_file": "raw/engine_latency_us.json"}
```
Bullet 3's `{D}` is `add_order.p99_us` from the T4 job (so the acceptance check's
"source real, GPU name contains T4" rule holds); the local machine's numbers are reported
beside it in BENCHMARKS.md from `results/engine-local-<stamp>/`.

## Dashboard export

`scripts/render_docs.py` also writes `public/data/gpu_lab.json` (ladder, roofline points,
precision table, slippage surface, OFI results and scatter sample, benchmark table with raw
links, provenance) and copies the pipeline timeline SVG to `public/data/pipeline_timeline.svg`.
Everything under `public/data/` must stay under 5 MB in total.
