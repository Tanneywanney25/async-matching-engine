# Benchmarks

Every figure on this page is rendered by `scripts/render_docs.py` from `results/*/metrics.json` of real-data runs on a Kaggle Tesla T4 and links to the raw samples it came from. Times are medians of interleaved timed iterations after discarded warmup runs, with the interquartile range (IQR) beside them; the SM clock, memory clock, temperature and power were sampled before and after every block.

## Hardware and data

| Item | Value |
|---|---|
| GPU | Tesla T4, 40 streaming multiprocessors, 15.6 GB, compute capability `7.5` |
| Theoretical memory bandwidth (from device properties) | 320.1 GB/s |
| Driver / CUDA runtime / nvcc | `580.178.04` / `12090` / `Cuda compilation tools, release 12.8, V12.8.93` |
| Nsight Systems | `NVIDIA Nsight Systems version 2025.1.1.0` |
| Host CPU | `Intel(R) Xeon(R) CPU @ 2.00GHz`, 4 logical cores |
| Run | `bench_rerun-20261021-191923` at git `efd6932b5833`, NVRTC target `sm_75` |
| Archive | 863,153 snapshots, 280,818 trades, source `real` |
| Capture window | `2026-10-21T07:49:24Z` to `2026-10-21T17:10:57Z` (9.36 hours), sequence gaps: 0 |
| Archive content hash | `10e3c7d79010feb6281d9c3833d00e6a05a51275dc9e1aa58c05c5aa75451959` |

## Memory bandwidth ceiling

| Measurement | Median ms (IQR) | GB/s (read + write) | Raw |
|---|---:|---:|---|
| Device-to-device copy kernel, 128-bit loads | 9.70 (IQR 0.0947) | 221.4 | [raw](results/bench_rerun-20261021-191923/raw/bandwidth__copy_kernel.json) |
| `cudaMemcpy` device to device | 9.25 (IQR 0.362) | 232.1 | [raw](results/bench_rerun-20261021-191923/raw/bandwidth__cudamemcpy_d2d.json) |
| Read-only streaming kernel, 128-bit loads (bytes read) | 4.09 (IQR 0.0172) | 262.7 | [raw](results/bench_rerun-20261021-191923/raw/bandwidth__read_kernel.json) |

A copy spends half its DRAM time writing and pays the read/write turnaround; the feature kernels read about seventeen bytes for every one they write, so the read-only figure is the fair ceiling for them and the highest measured figure is used.

Buffer size 1,073,741,824 bytes. Measured ceiling 262.7 GB/s, 82.1% of the 320.1 GB/s theoretical figure.

## Feature kernels: the optimization ladder

863,153 snapshots x 50 levels. Effective bandwidth counts the minimum traffic, 433 bytes per snapshot (every level's sizes, the top-of-book prices, the product id and six outputs). Per-kernel time is CUDA events around the launches only; end to end is wall clock including the host-to-device copy of exactly the inputs the kernels read and the copy back of the six outputs, from pinned host buffers, with the same from pageable NumPy memory beside it.

| Rung | Per-kernel median ms (IQR) | End to end ms | End to end, pageable ms | Effective GB/s | % of measured ceiling | % of theoretical | Block | Verified | Raw |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| v0 array-of-structures, one kernel per metric | 40.7 (IQR 24.7) | 101 | 202 | 9.2 | 3.5 | 2.9 | 256 | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_kernel__v0_aos.json) |
| v1 structure-of-arrays, one kernel per metric | 2.03 (IQR 0.0339) | 32.8 | 87.5 | 184.1 | 70.1 | 57.5 | 256 | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_kernel__v1_soa.json) |
| v2 structure-of-arrays, fused | 1.78 (IQR 0.0103) | 32.5 | 87.1 | 210.0 | 80.0 | 65.6 | 256 | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_kernel__v2_soa_fused.json) |
| v3 fused, 128-bit loads | 1.57 (IQR 0.0104) | 32.3 | 86.9 | 238.8 | 90.9 | 74.6 | 256 | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_kernel__v3_soa_fused_vec4.json) |
| v4 v3 + grid-stride, best launch configuration | 1.49 (IQR 0.0614) | 32.2 | 86.8 | 251.1 | 95.6 | 78.5 | 128 | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_kernel__v4_soa_fused_vec4_sweep.json) |

### Launch configuration sweep (v4, grid-stride)

| Block | Grid | Median ms | IQR ms | Effective GB/s |
|---:|---:|---:|---:|---:|
| 64 | 40 | 1.61 | 0.0181 | 231.7 |
| 64 | 80 | 1.53 | 0.0137 | 244.8 |
| 64 | 160 | 1.49 | 0.0125 | 250.9 |
| 64 | 320 | 1.64 | 0.0110 | 227.8 |
| 64 | 640 | 1.72 | 0.0118 | 217.8 |
| 64 | 1,280 | 1.70 | 0.00716 | 219.3 |
| 128 | 40 | 1.52 | 0.0154 | 246.0 |
| 128 | 80 | 1.48 | 0.0150 | 252.9 |
| 128 | 160 | 1.63 | 0.00693 | 229.9 |
| 128 | 320 | 1.69 | 0.0163 | 221.6 |
| 128 | 640 | 1.68 | 0.0126 | 222.7 |
| 128 | 1,280 | 1.65 | 0.00697 | 226.3 |
| 256 | 40 | 1.49 | 0.0138 | 251.3 |
| 256 | 80 | 1.63 | 0.00659 | 228.7 |
| 256 | 160 | 1.65 | 0.0114 | 226.9 |
| 256 | 320 | 1.62 | 0.0112 | 231.0 |
| 256 | 640 | 1.61 | 0.00885 | 232.5 |
| 256 | 1,280 | 1.59 | 0.0104 | 234.7 |
| 512 | 40 | 1.67 | 0.00995 | 224.4 |
| 512 | 80 | 1.60 | 0.00802 | 233.5 |
| 512 | 160 | 1.59 | 0.0159 | 234.6 |
| 512 | 320 | 1.56 | 0.0106 | 240.2 |
| 512 | 640 | 1.52 | 0.00598 | 245.2 |
| 512 | 1,280 | 1.53 | 0.00997 | 245.0 |
| 1,024 | 40 | 1.61 | 0.0219 | 232.3 |
| 1,024 | 80 | 1.55 | 0.00885 | 241.0 |
| 1,024 | 160 | 1.52 | 0.00452 | 245.8 |
| 1,024 | 320 | 1.50 | 0.00589 | 249.0 |
| 1,024 | 640 | 1.50 | 0.00803 | 248.9 |
| 1,024 | 1,280 | 1.50 | 0.00645 | 248.8 |

## Baselines (same machine, same job, same archive)

| Implementation | Median ms (IQR) | Verified | Raw |
|---|---:|:---:|---|
| C++ single thread, `g++ -O3 -march=native` | 38.4 (IQR 0.770) | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_e2e_and_cpu__cpp_single.json) |
| C++ with OpenMP on every core | 18.4 (IQR 0.323) | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_e2e_and_cpu__cpp_omp.json) |
| Vectorized NumPy | 104 (IQR 2.65) | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_e2e_and_cpu__numpy.json) |
| CuPy expression chain, per kernel | 4.53 (IQR 2.13) | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_kernel__cupy_chain.json) |
| CuPy expression chain, end to end | 94.5 (IQR 0.899) | yes | [raw](results/bench_rerun-20261021-191923/raw/ladder_e2e_and_cpu__cupy_e2e.json) |

Best rung: `v4_soa_fused_vec4_sweep`. Per kernel it is 25.8 times faster than single-threaded C++, 12.4 times faster than C++ with OpenMP on every core and 3.0 times faster than the CuPy chain. End to end, transfers included, it is 3.2 times faster than NumPy and 1.1 times faster than the CuPy chain with its own transfers. From pageable memory the end-to-end advantage over NumPy shrinks to 1.2 times.

## Prefix scans

### Archive-length series (863,153 float64 elements)

| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | Verified | Raw |
|---|---:|---:|---:|:---:|---|
| blelloch_shared | 0.244 (IQR 0.00690) | 3.54 | 56.7 | yes | [raw](results/bench_rerun-20261021-191923/raw/scan_snapshots__blelloch_shared.json) |
| warp_shuffle | 0.178 (IQR 0.00432) | 4.84 | 77.5 | yes | [raw](results/bench_rerun-20261021-191923/raw/scan_snapshots__warp_shuffle.json) |
| cub_device_scan | 0.120 (IQR 0.00529) | 7.17 | 114.7 | yes | [raw](results/bench_rerun-20261021-191923/raw/scan_snapshots__cub_device_scan.json) |

Winner: `cub_device_scan`.

### Large series (16,777,216 float64 elements)

| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | Verified | Raw |
|---|---:|---:|---:|:---:|---|
| blelloch_shared | 3.97 (IQR 0.00831) | 4.23 | 67.6 | yes | [raw](results/bench_rerun-20261021-191923/raw/scan_large__blelloch_shared.json) |
| warp_shuffle | 2.78 (IQR 0.00649) | 6.03 | 96.5 | yes | [raw](results/bench_rerun-20261021-191923/raw/scan_large__warp_shuffle.json) |
| cub_device_scan | 1.74 (IQR 0.0139) | 9.67 | 154.6 | yes | [raw](results/bench_rerun-20261021-191923/raw/scan_large__cub_device_scan.json) |

Winner: `cub_device_scan`.

### Rolling windows

- Rolling volume-weighted average price over 280,818 trades (100-trade window, and time windows sampled at 863,153 snapshots): 1.05 (IQR 0.00610) ms for the whole chain; maximum relative error against `metrics.rolling_vwap` in Decimal 1.9e-16 over 200 sampled trades. Prefixes are double-double; against `math.fsum` window sums at 2,000 sampled snapshots the one-second window's maximum relative error is 0.0e+00 (a plain float64 cumulative sum reaches 1.5e-05).
- Rolling one-second mean spread and imbalance over 863,153 snapshots: 1.22 (IQR 0.00630) ms, 706.2 million snapshots per second.

## Prefix-sum precision

Window sums of 100 values over 1,726,306 float32 top-of-book notionals, against an exact integer reference.

| Variant | Max relative error | Median relative error | Median ms (IQR) |
|---|---:|---:|---:|
| float32_naive | 3.5e+00 | 2.5e-04 | 0.426 (IQR 0.00615) |
| float64 | 1.8e-08 | 4.1e-15 | 0.445 (IQR 0.00376) |
| kahan_float32 | 1.3e-04 | 1.0e-08 | 1.06 (IQR 0.00779) |
| block_rebased_float32 | 4.7e-03 | 2.1e-07 | 0.512 (IQR 0.00174) |
| double_double (production) | 0.0e+00 | 0.0e+00 | 3.60 (IQR 0.00171) |

Production choice `double_double` costs 744.8% time against float32 naive.

## Layout under Nsight Systems

Total GPU time of the four book-level kernels over 30 timed passes plus the same warmup passes in both layouts, from `nsys stats --report cuda_gpu_kern_sum`.

| Kernel | Array-of-structures ms | Structure-of-arrays ms | Reduction % |
|---|---:|---:|---:|
| cumdepth | 3,648 | 142 | 96.1 |
| depth | 940 | 48.9 | 94.8 |
| imbalance | 171 | 9.85 | 94.3 |
| spread | 22.6 | 1.45 | 93.6 |
| **Total** | 4,782 | 202 | **95.8** |

Reports: [aos.nsys-rep](results/bench_all-20261021-190618/raw/profiles/aos.nsys-rep), [soa.nsys-rep](results/bench_all-20261021-190618/raw/profiles/soa.nsys-rep), [aos CSV](results/bench_all-20261021-190618/raw/profiles/aos_kern_sum.csv), [soa CSV](results/bench_all-20261021-190618/raw/profiles/soa_kern_sum.csv). Nsight Compute: blocked (`==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Perfor`).

## Execution-cost surface

110,483,584 simulated orders: 863,153 snapshots x both sides x 64 order sizes.

| Kernel | Median ms (IQR) | Orders per second (billions) | Raw |
|---|---:|---:|---|
| Naive, one thread per order | 285 (IQR 6.51) | 0.39 | [raw](results/bench_all-20261021-190618/raw/slippage_full_pass__naive.json) |
| Warp-cooperative | 65.8 (IQR 3.03) | 1.68 | [raw](results/bench_all-20261021-190618/raw/slippage_full_pass__warp.json) |

Warp-cooperative speedup 4.3 times; divergence penalty of the naive walk 76.9%. Verified against the Decimal walker on 1,000 snapshots: exact ticks `True`, maximum average-price error 2.1e-04 ticks. Surface: [per-minute medians](results/bench_all-20261021-190618/raw/slippage_surface.json).

## Order flow imbalance

10-level order flow imbalance (Cont, Kukanov and Stoikov), stencil kernel 2.05 (IQR 0.0588) ms. Out-of-sample fit on the last 30% of each product's time series:

| Product | Horizon | Out-of-sample R squared | In-sample R squared | Test points |
|---|---|---:|---:|---:|
| BTC-USD | 1 s | -0.0020 | 0.0084 | 131,142 |
| BTC-USD | 10 s | -0.0099 | 0.0073 | 131,099 |
| ETH-USD | 1 s | 0.0007 | 0.0069 | 127,798 |
| ETH-USD | 10 s | -0.0015 | 0.0051 | 127,759 |

A statistical description of the recorded window only; no trading or profit claim is made.

## End-to-end pipeline and transfers

16 chunks, pinned host memory, three streams.

| Measurement | Value |
|---|---:|
| Overlapped pass (wall clock) | 58.0 (IQR 0.0422) ms |
| Serial pass, one stream | 62.3 (IQR 0.0233) ms |
| Saved by overlap | 6.9% |
| Host to device | 57.7 ms at 12.0 GB/s |
| Kernels | 3.64 ms |
| Device to host | 1.70 ms at 10.2 GB/s |
| Transfer share of stage time | 94.2% |

Timeline: [SVG from the Nsight Systems trace](results/bench_all-20261021-190618/raw/pipeline_timeline.svg).

## Matching engine (CPU)

`scripts/benchmark.py`, 100,000 orders and 10,000 cancels, run on the T4 job's host CPU (`Intel(R) Xeon(R) CPU @ 2.00GHz`, Python `3.13.15`). Matching path only: an in-memory book, no feed, no network.

| Operation | Mean µs | p50 µs | p95 µs | p99 µs |
|---|---:|---:|---:|---:|
| `add_order` | 10.39 | 7.69 | 22.28 | 32.21 |
| `cancel_order` | 1.98 | 0.38 | 5.63 | 7.07 |

Raw latencies: [raw](results/bench_all-20261021-190618/raw/engine_latency_us.json). The laptop's own figures are in the README's Benchmark table.

## Clock throttling

No timed block lost more than a tenth of its SM clock between the samples taken before and after it.

## Runs

- [`bench_all-20261021-190618`](results/bench_all-20261021-190618/manifest.json)
- [`bench_rerun-20261021-191923`](results/bench_rerun-20261021-191923/manifest.json)
