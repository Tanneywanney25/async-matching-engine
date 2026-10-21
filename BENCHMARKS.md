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
| Run | `bench_rerun-20261021-232245` at git `d86371921eaf`, NVRTC target `sm_75` |
| Archive | 863,153 snapshots, 280,818 trades, source `real` |
| Capture window | `2026-10-21T07:49:24Z` to `2026-10-21T17:10:57Z` (9.36 hours), sequence gaps: 0 |
| Archive content hash | `10e3c7d79010feb6281d9c3833d00e6a05a51275dc9e1aa58c05c5aa75451959` |

## Memory bandwidth ceiling

| Measurement | Median ms (IQR) | GB/s (read + write) | Raw |
|---|---:|---:|---|
| Device-to-device copy kernel, 128-bit loads | 9.69 (IQR 0.0694) | 221.6 | [raw](results/bench_rerun-20261021-232245/raw/bandwidth__copy_kernel.json) |
| `cudaMemcpy` device to device | 9.21 (IQR 0.283) | 233.1 | [raw](results/bench_rerun-20261021-232245/raw/bandwidth__cudamemcpy_d2d.json) |
| Read-only streaming kernel, 128-bit loads (bytes read) | 4.09 (IQR 0.0130) | 262.4 | [raw](results/bench_rerun-20261021-232245/raw/bandwidth__read_kernel.json) |

A copy spends half its DRAM time writing and pays the read/write turnaround; the feature kernels read about seventeen bytes for every one they write, so the read-only figure is the fair ceiling for them and the highest measured figure is used.

Buffer size 1,073,741,824 bytes. Measured ceiling 262.4 GB/s, 82.0% of the 320.1 GB/s theoretical figure.

## Feature kernels: the optimization ladder

863,153 snapshots x 50 levels. Effective bandwidth counts the minimum traffic, 433 bytes per snapshot (every level's sizes, the top-of-book prices, the product id and six outputs). Per-kernel time is CUDA events around the launches only; end to end is wall clock including the host-to-device copy of exactly the inputs the kernels read and the copy back of the six outputs, from pinned host buffers, with the same from pageable NumPy memory beside it.

| Rung | Per-kernel median ms (IQR) | End to end ms | End to end, pageable ms | Effective GB/s | % of measured ceiling | % of theoretical | Block | Verified | Raw |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| v0 array-of-structures, one kernel per metric | 41.1 (IQR 0.378) | 100 | 194 | 9.1 | 3.5 | 2.8 | 256 | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_kernel__v0_aos.json) |
| v1 structure-of-arrays, one kernel per metric | 2.02 (IQR 0.00964) | 32.3 | 82.8 | 184.9 | 70.5 | 57.8 | 256 | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_kernel__v1_soa.json) |
| v2 structure-of-arrays, fused | 1.78 (IQR 0.00857) | 32.1 | 82.5 | 209.8 | 80.0 | 65.5 | 256 | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_kernel__v2_soa_fused.json) |
| v3 fused, 128-bit loads | 1.57 (IQR 0.00875) | 31.8 | 82.3 | 238.1 | 90.8 | 74.4 | 256 | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_kernel__v3_soa_fused_vec4.json) |
| v4 v3 + grid-stride, best launch configuration | 1.49 (IQR 0.00465) | 31.8 | 82.1 | 251.3 | 95.8 | 78.5 | 128 | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_kernel__v4_soa_fused_vec4_sweep.json) |

### Launch configuration sweep (v4, grid-stride)

| Block | Grid | Median ms | IQR ms | Effective GB/s |
|---:|---:|---:|---:|---:|
| 64 | 40 | 1.62 | 0.0165 | 230.1 |
| 64 | 80 | 1.53 | 0.0103 | 244.1 |
| 64 | 160 | 1.49 | 0.00819 | 250.8 |
| 64 | 320 | 1.64 | 0.00634 | 227.3 |
| 64 | 640 | 1.72 | 0.00686 | 217.8 |
| 64 | 1,280 | 1.71 | 0.00849 | 218.5 |
| 128 | 40 | 1.52 | 0.00890 | 245.6 |
| 128 | 80 | 1.48 | 0.0139 | 252.0 |
| 128 | 160 | 1.63 | 0.00670 | 229.8 |
| 128 | 320 | 1.69 | 0.0105 | 221.2 |
| 128 | 640 | 1.68 | 0.00842 | 222.0 |
| 128 | 1,280 | 1.65 | 0.00427 | 225.9 |
| 256 | 40 | 1.49 | 0.0109 | 250.9 |
| 256 | 80 | 1.64 | 0.00855 | 228.0 |
| 256 | 160 | 1.65 | 0.00995 | 227.0 |
| 256 | 320 | 1.62 | 0.00650 | 230.5 |
| 256 | 640 | 1.61 | 0.00997 | 231.5 |
| 256 | 1,280 | 1.60 | 0.00602 | 234.3 |
| 512 | 40 | 1.67 | 0.0112 | 224.1 |
| 512 | 80 | 1.60 | 0.00932 | 233.0 |
| 512 | 160 | 1.60 | 0.0179 | 234.2 |
| 512 | 320 | 1.56 | 0.0102 | 240.1 |
| 512 | 640 | 1.53 | 0.00599 | 244.7 |
| 512 | 1,280 | 1.53 | 0.00620 | 244.6 |
| 1,024 | 40 | 1.61 | 0.0127 | 232.1 |
| 1,024 | 80 | 1.55 | 0.00729 | 240.5 |
| 1,024 | 160 | 1.52 | 0.00885 | 245.2 |
| 1,024 | 320 | 1.50 | 0.00503 | 248.8 |
| 1,024 | 640 | 1.50 | 0.00665 | 248.6 |
| 1,024 | 1,280 | 1.50 | 0.00429 | 248.6 |

## Baselines (same machine, same job, same archive)

| Implementation | Median ms (IQR) | Verified | Raw |
|---|---:|:---:|---|
| C++ single thread, `g++ -O3 -march=native` | 34.9 (IQR 0.504) | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_e2e_and_cpu__cpp_single.json) |
| C++ with OpenMP on every core | 17.6 (IQR 0.664) | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_e2e_and_cpu__cpp_omp.json) |
| Vectorized NumPy | 90.1 (IQR 2.51) | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_e2e_and_cpu__numpy.json) |
| CuPy expression chain, per kernel | 4.28 (IQR 0.101) | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_kernel__cupy_chain.json) |
| CuPy expression chain, end to end | 90.7 (IQR 0.987) | yes | [raw](results/bench_rerun-20261021-232245/raw/ladder_e2e_and_cpu__cupy_e2e.json) |

Best rung: `v4_soa_fused_vec4_sweep`. Per kernel it is 23.5 times faster than single-threaded C++, 11.8 times faster than C++ with OpenMP on every core and 2.9 times faster than the CuPy chain. End to end, transfers included, it is 2.8 times faster than NumPy and 1.1 times faster than the CuPy chain with its own transfers. From pageable memory the end-to-end advantage over NumPy shrinks to 1.1 times.

## Prefix scans

GB/s counts every float64 element twice, once read and once written; the last column compares that with the read-only streaming ceiling measured in the same job.

Before it was timed, the single-pass decoupled look-back scan (Merrill and Garland, 2016) passed 16 correctness cases in a child process with a timeout, at sizes from 1 to 16,777,216 elements: 8 of them on integer-valued inputs, whose partial sums are exact in float64, so they must equal `numpy.cumsum` bit for bit whatever order the look-back adds the tiles in, and 20 repeated runs at 16,777,216 elements with 0 mismatches.

### Archive-length series (863,153 float64 elements)

| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | % of read ceiling | Verified | Raw |
|---|---:|---:|---:|---:|:---:|---|
| blelloch_shared | 0.246 (IQR 0.00376) | 3.50 | 56.1 | 21.4 | yes | [raw](results/bench_rerun-20261021-232245/raw/scan_snapshots__blelloch_shared.json) |
| warp_shuffle | 0.181 (IQR 0.00334) | 4.76 | 76.1 | 29.0 | yes | [raw](results/bench_rerun-20261021-232245/raw/scan_snapshots__warp_shuffle.json) |
| decoupled_lookback | 0.144 (IQR 0.00454) | 6.00 | 96.0 | 36.6 | yes | [raw](results/bench_rerun-20261021-232245/raw/scan_snapshots__decoupled_lookback.json) |
| cub_device_scan | 0.118 (IQR 0.00291) | 7.30 | 116.7 | 44.5 | yes | [raw](results/bench_rerun-20261021-232245/raw/scan_snapshots__cub_device_scan.json) |

Winner: `cub_device_scan`. The decoupled look-back scan ran in 0.144 ms against CUB's 0.118 ms: CUB is still faster, and the hand-written single pass is 21.5% slower.

### Large series (16,777,216 float64 elements)

| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | % of read ceiling | Verified | Raw |
|---|---:|---:|---:|---:|:---:|---|
| blelloch_shared | 3.97 (IQR 0.00448) | 4.23 | 67.6 | 25.8 | yes | [raw](results/bench_rerun-20261021-232245/raw/scan_large__blelloch_shared.json) |
| warp_shuffle | 2.78 (IQR 0.00670) | 6.03 | 96.4 | 36.8 | yes | [raw](results/bench_rerun-20261021-232245/raw/scan_large__warp_shuffle.json) |
| decoupled_lookback | 1.79 (IQR 0.0113) | 9.38 | 150.0 | 57.2 | yes | [raw](results/bench_rerun-20261021-232245/raw/scan_large__decoupled_lookback.json) |
| cub_device_scan | 1.73 (IQR 0.00933) | 9.69 | 155.1 | 59.1 | yes | [raw](results/bench_rerun-20261021-232245/raw/scan_large__cub_device_scan.json) |

Winner: `cub_device_scan`. The decoupled look-back scan ran in 1.79 ms against CUB's 1.73 ms: CUB is still faster, and the hand-written single pass is 3.4% slower.

### Rolling windows

- Rolling volume-weighted average price over 280,818 trades (100-trade window, and time windows sampled at 863,153 snapshots): 1.05 (IQR 0.00657) ms for the whole chain; maximum relative error against `metrics.rolling_vwap` in Decimal 1.9e-16 over 200 sampled trades. Prefixes are double-double; against `math.fsum` window sums at 2,000 sampled snapshots the one-second window's maximum relative error is 0.0e+00 (a plain float64 cumulative sum reaches 1.5e-05).
- Rolling one-second mean spread and imbalance over 863,153 snapshots: 1.21 (IQR 0.00466) ms, 712.5 million snapshots per second.

## Prefix-sum precision

Window sums of 100 values over 1,726,306 float32 top-of-book notionals, against an exact integer reference.

| Variant | Max relative error | Median relative error | Median ms (IQR) |
|---|---:|---:|---:|
| float32_naive | 3.5e+00 | 2.5e-04 | 0.406 (IQR 0.0509) |
| float64 | 1.8e-08 | 4.1e-15 | 0.432 (IQR 0.0310) |
| kahan_float32 | 1.3e-04 | 1.0e-08 | 0.926 (IQR 0.146) |
| block_rebased_float32 | 4.7e-03 | 2.1e-07 | 0.448 (IQR 0.0716) |
| double_double (production) | 0.0e+00 | 0.0e+00 | 3.11 (IQR 0.507) |

Production choice `double_double` costs 666.6% time against float32 naive.

## Layout under Nsight Systems

Total GPU time of the four book-level kernels over 30 timed passes plus the same warmup passes in both layouts, from `nsys stats --report cuda_gpu_kern_sum`.

| Kernel | Array-of-structures ms | Structure-of-arrays ms | Reduction % |
|---|---:|---:|---:|
| cumdepth | 3,650 | 140 | 96.2 |
| depth | 948 | 49.0 | 94.8 |
| imbalance | 175 | 9.85 | 94.4 |
| spread | 22.6 | 1.49 | 93.4 |
| **Total** | 4,796 | 200 | **95.8** |

Reports: [aos.nsys-rep](results/bench_all-20261021-230301/raw/profiles/aos.nsys-rep), [soa.nsys-rep](results/bench_all-20261021-230301/raw/profiles/soa.nsys-rep), [aos CSV](results/bench_all-20261021-230301/raw/profiles/aos_kern_sum.csv), [soa CSV](results/bench_all-20261021-230301/raw/profiles/soa_kern_sum.csv). Nsight Compute: blocked (`==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Perfor`).

## Execution-cost surface

110,483,584 simulated orders: 863,153 snapshots x both sides x 64 order sizes.

| Kernel | Median ms (IQR) | Orders per second (billions) | Raw |
|---|---:|---:|---|
| Naive, one thread per order | 286 (IQR 6.11) | 0.39 | [raw](results/bench_all-20261021-230301/raw/slippage_full_pass__naive.json) |
| Warp-cooperative | 65.4 (IQR 3.02) | 1.69 | [raw](results/bench_all-20261021-230301/raw/slippage_full_pass__warp.json) |

Warp-cooperative speedup 4.4 times; divergence penalty of the naive walk 77.1%. Verified against the Decimal walker on 1,000 snapshots: exact ticks `True`, maximum average-price error 2.1e-04 ticks. Surface: [per-minute medians](results/bench_all-20261021-230301/raw/slippage_surface.json).

## Order flow imbalance

10-level order flow imbalance (Cont, Kukanov and Stoikov), stencil kernel 2.07 (IQR 0.0771) ms. Out-of-sample fit on the last 30% of each product's time series:

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
| Overlapped pass (wall clock) | 57.2 (IQR 0.0572) ms |
| Serial pass, one stream | 61.5 (IQR 0.0419) ms |
| Saved by overlap | 7.0% |
| Host to device | 56.9 ms at 12.1 GB/s |
| Kernels | 3.66 ms |
| Device to host | 1.69 ms at 10.2 GB/s |
| Transfer share of stage time | 94.1% |

Timeline: [SVG from the Nsight Systems trace](results/bench_all-20261021-230301/raw/pipeline_timeline.svg).

## Matching engine (CPU)

`scripts/benchmark.py`, 100,000 orders and 10,000 cancels, run on the T4 job's host CPU (`Intel(R) Xeon(R) CPU @ 2.00GHz`, Python `3.13.15`). Matching path only: an in-memory book, no feed, no network.

| Operation | Mean µs | p50 µs | p95 µs | p99 µs |
|---|---:|---:|---:|---:|
| `add_order` | 11.05 | 8.19 | 23.90 | 34.57 |
| `cancel_order` | 2.29 | 0.46 | 6.34 | 9.34 |

Raw latencies: [raw](results/bench_all-20261021-230301/raw/engine_latency_us.json). The laptop's own figures are in the README's Benchmark table.

## Kernel resources and theoretical occupancy

Registers per thread, spills and static shared memory are `nvcc -cubin -arch=sm_75 -Xptxas -v` output (`Build cuda_12.8.r12.8/compiler.35583870_0`); blocks per streaming multiprocessor come from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at the block size each kernel was benchmarked with, recorded as the benchmarks ran. The card holds 65,536 registers, 1,024 threads, 65,536 bytes of shared memory and 16 resident blocks per streaming multiprocessor. Occupancy is the share of the multiprocessor's warp slots the resident blocks fill, and the last column names the resource that stops one more block from fitting, from a reimplementation of the occupancy calculator that agreed with the API on 47 of 47 launched configurations. For 33 of 43 kernels the NVRTC binary that was actually timed (CuPy's NVRTC `12.8`) has the same register and shared-memory counts as the nvcc build; where it differs the timed binary's count is shown in brackets, and in no case does that change the blocks per SM.

| Kernel | Source | Registers per thread | Spill stores / loads, bytes | Static shared memory, bytes | Block | Blocks per SM | Occupancy % | Limited by |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `copy_int4` | `copy.cu` | 40 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `read_int4` | `copy.cu` | 64 | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `dd_add_offsets` | `ddscan.cu` | 22 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `dd_block_offsets` | `ddscan.cu` | 26 | 0 / 0 | 0 | 1 | 16 | 50.0 | the 16-block cap |
| `dd_block_scan` | `ddscan.cu` | 38 (timed: 40) | 0 / 0 | 8,320 | 256 | 4 | 100.0 | warp slots |
| `window_diff_dd` | `ddscan.cu` | 16 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v0_depth` | `features.cu` | 48 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v0_imbalance` | `features.cu` | 44 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v0_microprice` | `features.cu` | 16 (timed: 15) | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v0_spread` | `features.cu` | 14 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v1_depth` | `features.cu` | 42 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v1_imbalance` | `features.cu` | 44 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v1_microprice` | `features.cu` | 17 (timed: 15) | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v1_spread` | `features.cu` | 12 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v2_fused` | `features.cu` | 48 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `v3_fused_vec4` | `features.cu` | 66 | 0 / 0 | 0 | 256 | 3 | 75.0 | registers |
| `v4_fused_vec4_gs` | `features.cu` | 64 | 0 / 0 | 0 | 128 | 8 | 100.0 | registers and warp slots |
| `layout_cumdepth_aos` | `layout.cu` | 64 | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `layout_cumdepth_soa` | `layout.cu` | 64 | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `layout_depth_aos` | `layout.cu` | 62 | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `layout_depth_soa` | `layout.cu` | 64 | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `layout_imbalance_aos` | `layout.cu` | 44 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `layout_imbalance_soa` | `layout.cu` | 44 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `layout_spread_aos` | `layout.cu` | 12 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `layout_spread_soa` | `layout.cu` | 8 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `ofi_stencil` | `ofi.cu` | 64 | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `pipeline_features_i32` | `pipeline.cu` | 51 (timed: 52) | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `block_local_scan_f32` | `precision.cu` | 22 (timed: 25) | 0 / 0 | 4,128 | 256 | 4 | 100.0 | warp slots |
| `block_totals_f64` | `precision.cu` | 22 | 0 / 0 | 64 | 256 | 4 | 100.0 | warp slots |
| `kahan_chunks_f32` | `precision.cu` | 54 | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `window_diff_f64` | `precision.cu` | 10 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `window_diff_local` | `precision.cu` | 30 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `book_features_sorted` | `rolling.cu` | 50 (timed: 46) | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
| `vwap_time_window_dd` | `rolling.cu` | 26 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `vwap_trade_window_dd` | `rolling.cu` | 26 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `window_mean` | `rolling.cu` | 26 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `blelloch_f64` | `scan.cu` | 30 (timed: 29) | 0 / 0 | 8,712 | 512 | 2 | 100.0 | warp slots |
| `scan_add_offsets_f32` | `scan.cu` | 12 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `scan_add_offsets_f64` | `scan.cu` | 14 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `warpscan_f32` | `scan.cu` | 26 (timed: 23) | 0 / 0 | 4,128 | 256 | 4 | 100.0 | warp slots |
| `warpscan_f64` | `scan.cu` | 28 (timed: 30) | 0 / 0 | 8,256 | 256 | 4 | 100.0 | warp slots |
| `slippage_naive` | `slippage.cu` | 38 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `slippage_warp` | `slippage.cu` | 49 (timed: 48) | 0 / 0 | 10,240 | 256 | 4 | 100.0 | registers and warp slots |
| CUB scan kernel at 964 tiles of 128 x 7 elements | `libscan_cub.so` | 49 | stack 0 | 7,184 | 128 | 8 | 100.0 | shared memory and warp slots |
| CUB scan kernel at 18,725 tiles of 128 x 7 elements | `libscan_cub.so` | 49 | stack 0 | 7,184 | 128 | 8 | 100.0 | shared memory and warp slots |

CUB's kernels are compiled into `libscan_cub.so`, so their registers and shared memory come from `cuobjdump --dump-resource-usage` (which reports the stack frame rather than spills) and their launch shape and blocks per SM from CUB's own `-DCUB_DEBUG_LOG` output, which computes occupancy with the same API. Raw: [ptxas logs and CUB log](results/bench_all-20261021-230301/raw/resources), [launch configurations](results/bench_all-20261021-230301/raw/resources_launches.json).

### Occupancy across the v4 launch sweep

| Block | Blocks per SM | Occupancy % | Limited by | Fastest grid's median ms |
|---:|---:|---:|---|---:|
| 64 | 16 | 100.0 | registers, warp slots and the 16-block cap | 1.49 |
| 128 | 8 | 100.0 | registers and warp slots | 1.48 |
| 256 | 4 | 100.0 | registers and warp slots | 1.49 |
| 512 | 2 | 100.0 | registers and warp slots | 1.53 |
| 1,024 | 1 | 100.0 | registers and warp slots | 1.50 |

Every swept block size reaches the same theoretical occupancy, so occupancy does not explain the differences across the sweep; the grid size changes them more (see the launch configuration sweep above).

## Clock throttling

No timed block lost more than a tenth of its SM clock between the samples taken before and after it, on every card the block used.

## Runs

- [`bench_all-20261021-230301`](results/bench_all-20261021-230301/manifest.json)
- [`bench_rerun-20261021-232245`](results/bench_rerun-20261021-232245/manifest.json)
