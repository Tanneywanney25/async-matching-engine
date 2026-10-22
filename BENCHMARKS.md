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
| Run | `bench_all-20261022-011024` at git `af78a2c85fd2`, NVRTC target `sm_75` |
| Archive | 863,153 snapshots, 280,818 trades, source `real` |
| Capture window | `2026-10-21T07:49:24Z` to `2026-10-21T17:10:57Z` (9.36 hours), sequence gaps: 0 |
| Archive content hash | `10e3c7d79010feb6281d9c3833d00e6a05a51275dc9e1aa58c05c5aa75451959` |

## Memory bandwidth ceiling

| Measurement | Median ms (IQR) | GB/s (read + write) | Raw |
|---|---:|---:|---|
| Device-to-device copy kernel, 128-bit loads | 9.69 (IQR 0.0970) | 221.6 | [raw](results/bench_all-20261022-011024/raw/bandwidth__copy_kernel.json) |
| `cudaMemcpy` device to device | 9.26 (IQR 0.299) | 231.9 | [raw](results/bench_all-20261022-011024/raw/bandwidth__cudamemcpy_d2d.json) |
| Read-only streaming kernel, 128-bit loads (bytes read) | 4.09 (IQR 0.0213) | 262.3 | [raw](results/bench_all-20261022-011024/raw/bandwidth__read_kernel.json) |

A copy spends half its DRAM time writing and pays the read/write turnaround; the feature kernels read about seventeen bytes for every one they write, so the read-only figure is the fair ceiling for them and the highest measured figure is used.

Buffer size 1,073,741,824 bytes. Measured ceiling 262.3 GB/s, 82.0% of the 320.1 GB/s theoretical figure.

## Feature kernels: the optimization ladder

863,153 snapshots x 50 levels. Effective bandwidth counts the minimum traffic, 433 bytes per snapshot (every level's sizes, the top-of-book prices, the product id and six outputs). Per-kernel time is CUDA events around the launches only; end to end is wall clock including the host-to-device copy of exactly the inputs the kernels read and the copy back of the six outputs, from pinned host buffers, with the same from pageable NumPy memory beside it.

| Rung | Per-kernel median ms (IQR) | End to end ms | End to end, pageable ms | Effective GB/s | % of measured ceiling | % of theoretical | Block | Verified | Raw |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| v0 array-of-structures, one kernel per metric | 41.9 (IQR 26.3) | 100 | 192 | 8.9 | 3.4 | 2.8 | 256 | yes | [raw](results/bench_all-20261022-011024/raw/ladder_kernel__v0_aos.json) |
| v1 structure-of-arrays, one kernel per metric | 2.02 (IQR 0.0143) | 32.3 | 81.1 | 184.6 | 70.4 | 57.7 | 256 | yes | [raw](results/bench_all-20261022-011024/raw/ladder_kernel__v1_soa.json) |
| v2 structure-of-arrays, fused | 1.79 (IQR 0.0104) | 32.1 | 80.6 | 209.4 | 79.8 | 65.4 | 256 | yes | [raw](results/bench_all-20261022-011024/raw/ladder_kernel__v2_soa_fused.json) |
| v3 fused, 128-bit loads | 1.57 (IQR 0.00834) | 31.9 | 80.6 | 238.3 | 90.8 | 74.4 | 256 | yes | [raw](results/bench_all-20261022-011024/raw/ladder_kernel__v3_soa_fused_vec4.json) |
| v4 v3 + grid-stride, best launch configuration | 1.49 (IQR 0.0521) | 31.8 | 80.4 | 251.1 | 95.7 | 78.5 | 128 | yes | [raw](results/bench_all-20261022-011024/raw/ladder_kernel__v4_soa_fused_vec4_sweep.json) |
| Triton, fused like v2 (extra rung, best of a small sweep) | 1.76 (IQR 0.0160) | 32.0 | 80.8 | 212.7 | 81.1 | 66.4 | 1,024 | yes | [raw](results/bench_all-20261022-011024/raw/ladder_kernel__triton_fused.json) |

### Launch configuration sweep (v4, grid-stride)

| Block | Grid | Median ms | IQR ms | Effective GB/s |
|---:|---:|---:|---:|---:|
| 64 | 40 | 1.62 | 0.0135 | 231.0 |
| 64 | 80 | 1.53 | 0.0116 | 245.0 |
| 64 | 160 | 1.49 | 0.0175 | 250.5 |
| 64 | 320 | 1.64 | 0.0109 | 228.4 |
| 64 | 640 | 1.72 | 0.0127 | 217.5 |
| 64 | 1,280 | 1.71 | 0.0112 | 218.5 |
| 128 | 40 | 1.52 | 0.00758 | 246.3 |
| 128 | 80 | 1.48 | 0.0113 | 252.4 |
| 128 | 160 | 1.62 | 0.0102 | 230.3 |
| 128 | 320 | 1.69 | 0.00877 | 220.9 |
| 128 | 640 | 1.68 | 0.00592 | 222.2 |
| 128 | 1,280 | 1.66 | 0.00753 | 225.6 |
| 256 | 40 | 1.50 | 0.0200 | 249.6 |
| 256 | 80 | 1.63 | 0.00949 | 228.6 |
| 256 | 160 | 1.65 | 0.00998 | 226.2 |
| 256 | 320 | 1.63 | 0.0121 | 230.0 |
| 256 | 640 | 1.62 | 0.00778 | 231.3 |
| 256 | 1,280 | 1.60 | 0.0128 | 234.2 |
| 512 | 40 | 1.67 | 0.0126 | 224.4 |
| 512 | 80 | 1.60 | 0.0108 | 233.1 |
| 512 | 160 | 1.59 | 0.0107 | 234.4 |
| 512 | 320 | 1.56 | 0.00874 | 239.5 |
| 512 | 640 | 1.53 | 0.00589 | 244.5 |
| 512 | 1,280 | 1.53 | 0.0128 | 244.2 |
| 1,024 | 40 | 1.61 | 0.0161 | 232.0 |
| 1,024 | 80 | 1.55 | 0.0120 | 240.7 |
| 1,024 | 160 | 1.53 | 0.00838 | 245.0 |
| 1,024 | 320 | 1.51 | 0.00570 | 248.3 |
| 1,024 | 640 | 1.50 | 0.00315 | 248.5 |
| 1,024 | 1,280 | 1.50 | 0.00564 | 248.6 |

### v0 and the card's clock

v0 is the one rung whose time changes with the card's state. Its 30 interleaved samples fall in two levels: 8 (the first ones) near 10.6 ms and 22 near 42.0 ms. In the background log, sampled every 100 ms, the fast ones fall at an SM clock of at most 675 MHz, before the card boosted, and the slow ones at a median of 1,395 MHz, 21 of them with the software power cap active. Timed alone, v0 took 41.9 ms at the median (IQR 0.163) with the card at 1,350 to 1,590 MHz, so the slow level is its steady state once the card is under sustained load, the state the table's median reflects. Over the same iterations the other variants changed between 32.4% faster (`cupy_chain`) and 0.7% slower (`v1_soa`). Why the uncoalesced array-of-structures kernels slow down at the higher clock is not established by these measurements.

### GPU time with the launches queued ahead

The per-kernel times above record the start event on an idle GPU, so each includes the host's dispatch of the launch. This block, interleaved like the others, queues each pass behind a 1,000 µs busy-wait kernel, so the events bracket GPU execution only, and times how long the host took to enqueue the pass. The slowest enqueue took 313.69 µs, well inside the wait.

| Rung | Events around the launch, median ms | Launches queued, median ms (IQR) | Host enqueue, median µs |
|---|---:|---:|---:|
| v0 array-of-structures, one kernel per metric | 41.9 | 42.2 (IQR 0.179) | 51.33 |
| v1 structure-of-arrays, one kernel per metric | 2.02 | 2.00 (IQR 0.0101) | 58.20 |
| v2 structure-of-arrays, fused | 1.79 | 1.77 (IQR 0.00899) | 20.87 |
| v3 fused, 128-bit loads | 1.57 | 1.55 (IQR 0.00735) | 20.31 |
| v4 v3 + grid-stride, best launch configuration | 1.49 | 1.47 (IQR 0.0107) | 17.71 |
| Triton, fused like v2 (extra rung, best of a small sweep) | 1.76 | 1.69 (IQR 0.0106) | 65.04 |

Where the queued median sits above the event-timed one (`v0_aos`), the two blocks ran at different times, and the difference is not dispatch: it is the block-to-block variation of the card's clock and power state, which v0 feels most (see above).

### Extra rung: the fused kernel in Triton

Triton `3.6.0` compiles the same fused computation as v2 (level sums in level order, IEEE division) and reads the ladder's own device arrays, so it competes on identical data. Every launch configuration was checked against the NumPy reference before any was timed; the ones that passed, timed with launches queued ahead, GPU time only, in an interleaved block of their own: 256 snapshots and 4 warps per program: 1.81 ms; 512 snapshots and 4 warps per program: 1.75 ms; 1,024 snapshots and 4 warps per program: 1.73 ms; 1,024 snapshots and 8 warps per program: 1.70 ms. Measured with its launch queued ahead, the best one takes 0.96 times the GPU time of the CUDA v2 kernel it mirrors and 1.15 times that of the best CUDA rung (v4 v3 + grid-stride, best launch configuration). With the events around the launch, as in the ladder table, the ratios are 0.98 and 1.18: enqueuing one Triton launch took a median 65.04 µs on the host, against 20.87 µs for v2 through CuPy. It sits in the ladder table (its Block column is snapshots per program) and is never counted as the best rung. Its outputs are bit-for-bit identical to v2's.

## Baselines (same machine, same job, same archive)

| Implementation | Median ms (IQR) | Verified | Raw |
|---|---:|:---:|---|
| C++ single thread, `g++ -O3 -march=native` | 35.0 (IQR 0.759) | yes | [raw](results/bench_all-20261022-011024/raw/ladder_e2e_and_cpu__cpp_single.json) |
| C++ with OpenMP on every core | 17.2 (IQR 0.403) | yes | [raw](results/bench_all-20261022-011024/raw/ladder_e2e_and_cpu__cpp_omp.json) |
| Vectorized NumPy | 84.8 (IQR 2.29) | yes | [raw](results/bench_all-20261022-011024/raw/ladder_e2e_and_cpu__numpy.json) |
| CuPy expression chain, per kernel | 4.16 (IQR 1.99) | yes | [raw](results/bench_all-20261022-011024/raw/ladder_kernel__cupy_chain.json) |
| CuPy expression chain, end to end | 88.9 (IQR 0.584) | yes | [raw](results/bench_all-20261022-011024/raw/ladder_e2e_and_cpu__cupy_e2e.json) |

Best rung: `v4_soa_fused_vec4_sweep`. Per kernel it is 23.5 times faster than single-threaded C++, 11.5 times faster than C++ with OpenMP on every core and 2.8 times faster than the CuPy chain. End to end, transfers included, it is 2.7 times faster than NumPy and 1.1 times faster than the CuPy chain with its own transfers. From pageable memory the end-to-end advantage over NumPy shrinks to 1.1 times.

## Prefix scans

GB/s counts every float64 element twice, once read and once written; the last column compares that with the read-only streaming ceiling measured in the same job.

Before it was timed, the single-pass decoupled look-back scan (Merrill and Garland, 2016) passed 16 correctness cases in a child process with a timeout, at sizes from 1 to 16,777,216 elements: 8 of them on integer-valued inputs, whose partial sums are exact in float64, so they must equal `numpy.cumsum` bit for bit whatever order the look-back adds the tiles in, and 20 repeated runs at 16,777,216 elements with 0 mismatches.

### Archive-length series (863,153 float64 elements)

| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | % of read ceiling | Verified | Raw |
|---|---:|---:|---:|---:|:---:|---|
| blelloch_shared | 0.248 (IQR 0.00718) | 3.48 | 55.6 | 21.2 | yes | [raw](results/bench_all-20261022-011024/raw/scan_snapshots__blelloch_shared.json) |
| warp_shuffle | 0.182 (IQR 0.00404) | 4.74 | 75.8 | 28.9 | yes | [raw](results/bench_all-20261022-011024/raw/scan_snapshots__warp_shuffle.json) |
| decoupled_lookback | 0.143 (IQR 0.00482) | 6.02 | 96.2 | 36.7 | yes | [raw](results/bench_all-20261022-011024/raw/scan_snapshots__decoupled_lookback.json) |
| cub_device_scan | 0.119 (IQR 0.00304) | 7.27 | 116.2 | 44.3 | yes | [raw](results/bench_all-20261022-011024/raw/scan_snapshots__cub_device_scan.json) |

Winner: `cub_device_scan`. The decoupled look-back scan ran in 0.143 ms against CUB's 0.119 ms: CUB is still faster, and the hand-written single pass is 20.8% slower.

### Large series (16,777,216 float64 elements)

| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | % of read ceiling | Verified | Raw |
|---|---:|---:|---:|---:|:---:|---|
| blelloch_shared | 3.97 (IQR 0.0102) | 4.22 | 67.6 | 25.8 | yes | [raw](results/bench_all-20261022-011024/raw/scan_large__blelloch_shared.json) |
| warp_shuffle | 2.79 (IQR 0.0109) | 6.02 | 96.3 | 36.7 | yes | [raw](results/bench_all-20261022-011024/raw/scan_large__warp_shuffle.json) |
| decoupled_lookback | 1.79 (IQR 0.0118) | 9.37 | 149.9 | 57.1 | yes | [raw](results/bench_all-20261022-011024/raw/scan_large__decoupled_lookback.json) |
| cub_device_scan | 1.73 (IQR 0.00714) | 9.70 | 155.1 | 59.1 | yes | [raw](results/bench_all-20261022-011024/raw/scan_large__cub_device_scan.json) |

Winner: `cub_device_scan`. The decoupled look-back scan ran in 1.79 ms against CUB's 1.73 ms: CUB is still faster, and the hand-written single pass is 3.5% slower.

### Rolling windows

- Rolling volume-weighted average price over 280,818 trades (100-trade window, and time windows sampled at 863,153 snapshots): 1.06 (IQR 0.00757) ms for the whole chain; maximum relative error against `metrics.rolling_vwap` in Decimal 1.9e-16 over 200 sampled trades. Prefixes are double-double; against `math.fsum` window sums at 2,000 sampled snapshots the one-second window's maximum relative error is 0.0e+00 (a plain float64 cumulative sum reaches 1.5e-05).
- Rolling one-second mean spread and imbalance over 863,153 snapshots: 1.46 (IQR 0.00457) ms, 590.5 million snapshots per second.

## Prefix-sum precision

Window sums of 100 values over 1,726,306 float32 top-of-book notionals, against an exact integer reference.

| Variant | Max relative error | Median relative error | Median ms (IQR) |
|---|---:|---:|---:|
| float32_naive | 3.5e+00 | 2.5e-04 | 0.430 (IQR 0.0837) |
| float64 | 1.8e-08 | 4.1e-15 | 0.447 (IQR 0.0440) |
| kahan_float32 | 1.3e-04 | 1.0e-08 | 0.962 (IQR 0.231) |
| block_rebased_float32 | 4.7e-03 | 2.1e-07 | 0.494 (IQR 0.105) |
| double_double (production) | 0.0e+00 | 0.0e+00 | 3.21 (IQR 0.830) |

Production choice `double_double` costs 645.4% time against float32 naive.

## Layout under Nsight Systems

Total GPU time of the four book-level kernels over 30 timed passes plus the same warmup passes in both layouts, from `nsys stats --report cuda_gpu_kern_sum`.

| Kernel | Array-of-structures ms | Structure-of-arrays ms | Reduction % |
|---|---:|---:|---:|
| cumdepth | 3,658 | 144 | 96.1 |
| depth | 968 | 49.0 | 94.9 |
| imbalance | 199 | 9.84 | 95.0 |
| spread | 22.7 | 1.45 | 93.6 |
| **Total** | 4,848 | 204 | **95.8** |

Reports: [aos.nsys-rep](results/bench_all-20261022-011024/raw/profiles/aos.nsys-rep), [soa.nsys-rep](results/bench_all-20261022-011024/raw/profiles/soa.nsys-rep), [aos CSV](results/bench_all-20261022-011024/raw/profiles/aos_kern_sum.csv), [soa CSV](results/bench_all-20261022-011024/raw/profiles/soa_kern_sum.csv). Nsight Compute: blocked (`==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Perfor`).

## Execution-cost surface

110,483,584 simulated orders: 863,153 snapshots x both sides x 64 order sizes.

| Kernel | Median ms (IQR) | Orders per second (billions) | Raw |
|---|---:|---:|---|
| Naive, one thread per order | 282 (IQR 1.92) | 0.39 | [raw](results/bench_all-20261022-011024/raw/slippage_full_pass__naive.json) |
| Warp-cooperative | 61.9 (IQR 1.17) | 1.78 | [raw](results/bench_all-20261022-011024/raw/slippage_full_pass__warp.json) |

Warp-cooperative speedup 4.6 times; divergence penalty of the naive walk 78.0%. Verified against the Decimal walker on 1,000 snapshots: exact ticks `True`, maximum average-price error 2.1e-04 ticks. Surface: [per-minute medians](results/bench_all-20261022-011024/raw/slippage_surface.json).

## Order flow imbalance

10-level order flow imbalance (Cont, Kukanov and Stoikov), stencil kernel 2.08 (IQR 0.0741) ms. Out-of-sample fit on the last 30% of each product's time series:

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
| Overlapped pass (wall clock) | 57.7 (IQR 0.0370) ms |
| Serial pass, one stream | 62.0 (IQR 0.0387) ms |
| Saved by overlap | 6.9% |
| Host to device | 57.3 ms at 12.0 GB/s |
| Kernels | 3.64 ms |
| Device to host | 1.71 ms at 10.1 GB/s |
| Transfer share of stage time | 94.2% |

Timeline: [SVG from the Nsight Systems trace](results/bench_all-20261022-011024/raw/pipeline_timeline.svg).

### Narrower prices on the wire: int16 tick offsets

Each side's best price travels as int32 per snapshot and every level as an int16 offset from it, decoded inside the kernel; a chunk holding any offset beyond 32,767 ticks is sent as int32 instead.

| Measurement | int32 | int16 offsets |
|---|---:|---:|
| Bytes host to device per pass | 690,536,448 | 524,812,288 |
| End to end, overlapped (median ms, IQR) | 57.6 (IQR 0.0269) | 43.2 (IQR 0.0435) |
| Host to device, summed over chunks (ms) | 57.3 | 42.9 |

The encoding moves 24.0% fewer bytes; none of the 16 chunks needed the int32 fallback, because every recorded level sat within 32,767 ticks of its side's best price. End to end the pass is 25.1% faster, and its outputs are bit-for-bit identical to the int32 pass's (verified against the NumPy reference too).

### One card against two

`nvidia-smi topo -m` reports the two cards as `PHB`, connected through a PCIe host bridge, that is the CPU; each has its own PCIe link (`GPU0` at `00000000:00:04.0`, rated `Gen3 x16` and running at `Gen3 x16` just after the copies, `GPU1` at `00000000:00:05.0`, rated `Gen3 x16` and running at `Gen3 x16` just after the copies). The archive is split in two contiguous halves of 8 chunks, each card with its own pinned host buffer, module, streams and device buffers, and the two cards' chunks are issued alternately.

| Measurement | One card | Two cards |
|---|---:|---:|
| Whole pipeline, overlapped (median ms, IQR) | 57.7 (IQR 0.0389) | 56.3 (IQR 0.0231) |
| Copy-only pass, host to device (median ms, IQR) | 57.1 (IQR 0.0241) | 56.0 (IQR 0.00925) |
| Host-to-device bandwidth (GB/s, aggregate for two) | 12.1 | 12.3 |

Two cards take 56.3 ms against 57.7 ms on one, a speedup of 1.02 (51.2% of perfect scaling), and copy 1.02 times as many bytes per second as one card: two cards together copy no faster than one, so the limit sits upstream of the cards' own links, on a path both share, and the second card only adds compute this transfer-bound pipeline does not need. A single-threaded host memcpy on this machine runs at 5.5 GB/s for comparison. The two-card outputs are bit-for-bit identical to the one-card outputs and match the NumPy reference. The result is not stable across Kaggle sessions. Of the 3 full runs that measured it, each in its own session, `bench_rerun-20261021-233009` found that two cards copied no faster than one; `bench_all-20261022-003544` found that two cards copied close to twice as fast as one; `bench_all-20261022-011024` found that two cards copied no faster than one, and their `nvidia-smi` topology, bus ids and rated links read the same. These runs do not show why; this page reports `bench_all-20261022-011024`, the latest.

## Matching engine (CPU)

`scripts/benchmark.py`, 100,000 orders and 10,000 cancels, run on the T4 job's host CPU (`Intel(R) Xeon(R) CPU @ 2.00GHz`, Python `3.13.15`). Matching path only: an in-memory book, no feed, no network.

| Operation | Mean µs | p50 µs | p95 µs | p99 µs |
|---|---:|---:|---:|---:|
| `add_order` | 10.31 | 7.65 | 22.27 | 31.82 |
| `cancel_order` | 1.80 | 0.36 | 5.04 | 5.96 |

Raw latencies: [raw](results/bench_all-20261022-011024/raw/engine_latency_us.json). The laptop's own figures are in the README's Benchmark table.

## C++20 order book core (CPU)

`native/`, built with CMake in the same Kaggle job (`c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, -O3 (CMake Release), C++20) and run on its host CPU (`Intel(R) Xeon(R) CPU @ 2.00GHz`). Its native tests passed 2,482,227 checks, among them a randomized differential test against a `std::map` and `std::deque` reference book, and a counting allocator saw 0 heap allocations over 200,000 operations once the book was built. The differential fuzzer ran 10,000 seeded streams (548,317 operations) against the reference book with 0 divergences, comparing fills, statuses, best prices and depth after every operation.

Stream: scripts/benchmark.py's stream: 100,000 limit orders, price gauss(mid, 1% of mid) to the cent, size uniform(0.001, 2.0) to 1e-6, seed 42; 10,000 cancels of rested ids. Every path below replays the same operations in one process; they agreed on fills, resting orders and cancels found: yes.

| Path | add p50 µs | add p99 µs | add p99.9 µs | cancel p50 µs | cancel p99 µs | cancel p99.9 µs |
|---|---:|---:|---:|---:|---:|---:|
| C++ core, native harness (no Python in the loop) | 0.077 | 0.386 | 0.575 | 0.048 | 0.400 | 0.543 |
| C++ core through the pybind11 binding, integer ticks and lots | 0.508 | 1.079 | 4.472 | 0.391 | 0.930 | 3.004 |
| C++ core through `NativeOrderBook`, Decimal orders | 3.428 | 15.0 | 29.8 | 0.530 | 1.760 | 5.249 |
| Python engine (the reference implementation) | 6.208 | 28.3 | 57.7 | 0.428 | 6.102 | 15.6 |

Of the 10,000 cancels, 3,575 found a resting order; the others name orders the stream had already filled, so their latency is a lookup that finds nothing. Every ratio here uses the Python engine row of this table, measured in the same process on the same stream (`scripts/benchmark.py`, run separately in this job, reported a p99 `add_order` of 31.82 µs). At p99 the native `add` is 73.4 times faster than the Python engine's, 26.3 times faster through the binding with integers and 1.9 times faster with Decimal orders. The integer binding row adds the call from Python to the native work; the Decimal row adds, in `NativeOrderBook`, the conversion to ticks and lots and the Python `Fill` objects built for every fill. The native harness reads `std::chrono::steady_clock` twice per operation; a pair of reads costs 0.024 µs at the median, included above, not subtracted. Raw: [raw](results/bench_all-20261022-011024/raw/synthetic_native_engine_latency_us.json).

### The same paths on a stream derived from the recorded book

Stream: derived from the first 95.5 minutes of the recorded BTC-USD book: 82,059 snapshots reconciled into limit orders and cancels, 26,311 trades replayed as market orders on the aggressor's side (1,000,020 operations: 491,040 limit orders, 26,311 market orders, 482,669 cancels). Removals come before additions within each snapshot, and no limit order matched on arrival, so every fill comes from a market order. The paths agreed on fills, resting orders and cancels found: yes.

| Path | limit p50 µs | limit p99 µs | market p50 µs | market p99 µs | cancel p50 µs | cancel p99 µs |
|---|---:|---:|---:|---:|---:|---:|
| C++ core, native harness (no Python in the loop) | 0.062 | 0.140 | 0.038 | 0.203 | 0.050 | 0.098 |
| C++ core through the pybind11 binding, integer ticks and lots | 0.542 | 0.969 | 0.618 | 3.437 | 0.367 | 0.712 |
| C++ core through `NativeOrderBook`, Decimal orders | 2.941 | 6.386 | 4.359 | 17.7 | 0.572 | 1.004 |
| Python engine (the reference implementation) | 5.799 | 12.7 | 6.419 | 28.3 | 2.395 | 5.462 |

Against the Python engine row, measured in the same process on the same operations, the native core is 90.6 times faster at p99 on limit orders and 139 times faster on market orders; on cancels it is 55.7 times faster. On the synthetic stream above the `add` ratio is 73.4. A pair of clock reads in this mode costs 0.024 µs at the median, included above. Raw: [raw](results/bench_all-20261022-011024/raw/realistic_native_engine_latency_us.json).

## Kernel resources and theoretical occupancy

Registers per thread, spills and static shared memory are `nvcc -cubin -arch=sm_75 -Xptxas -v` output (`Build cuda_12.8.r12.8/compiler.35583870_0`); blocks per streaming multiprocessor come from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at the block size each kernel was benchmarked with, recorded as the benchmarks ran. The card holds 65,536 registers, 1,024 threads, 65,536 bytes of shared memory and 16 resident blocks per streaming multiprocessor. Occupancy is the share of the multiprocessor's warp slots the resident blocks fill, and the last column names the resource that stops one more block from fitting, from a reimplementation of the occupancy calculator that agreed with the API on 50 of 50 launched configurations. For 34 of 46 kernels the NVRTC binary that was actually timed (CuPy's NVRTC `12.8`) has the same register and shared-memory counts as the nvcc build; where it differs the timed binary's count is shown in brackets, and in no case does that change the blocks per SM. Not listed: the Triton rung's kernel, which Triton compiles rather than nvcc, and the one-thread busy-wait the queued timing launches ahead of each pass.

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
| `pipeline_features_i16` | `pipeline.cu` | 50 (timed: 48) | 0 / 0 | 0 | 256 | 4 | 100.0 | registers and warp slots |
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
| `lookback_f64` | `scan.cu` | 48 (timed: 44) | 0 / 0 | 18,508 | 256 | 3 | 75.0 | shared memory |
| `lookback_init` | `scan.cu` | 26 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `scan_add_offsets_f32` | `scan.cu` | 12 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `scan_add_offsets_f64` | `scan.cu` | 14 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `warpscan_f32` | `scan.cu` | 26 (timed: 23) | 0 / 0 | 4,128 | 256 | 4 | 100.0 | warp slots |
| `warpscan_f64` | `scan.cu` | 28 (timed: 30) | 0 / 0 | 8,256 | 256 | 4 | 100.0 | warp slots |
| `slippage_naive` | `slippage.cu` | 38 | 0 / 0 | 0 | 256 | 4 | 100.0 | warp slots |
| `slippage_warp` | `slippage.cu` | 49 (timed: 48) | 0 / 0 | 10,240 | 256 | 4 | 100.0 | registers and warp slots |
| CUB scan kernel at 964 tiles of 128 x 7 elements | `libscan_cub.so` | 49 | stack 0 | 7,184 | 128 | 8 | 100.0 | shared memory and warp slots |
| CUB scan kernel at 18,725 tiles of 128 x 7 elements | `libscan_cub.so` | 49 | stack 0 | 7,184 | 128 | 8 | 100.0 | shared memory and warp slots |

CUB's kernels are compiled into `libscan_cub.so`, so their registers and shared memory come from `cuobjdump --dump-resource-usage` (which reports the stack frame rather than spills) and their launch shape and blocks per SM from CUB's own `-DCUB_DEBUG_LOG` output, which computes occupancy with the same API. Raw: [ptxas logs and CUB log](results/bench_all-20261022-011024/raw/resources), [launch configurations](results/bench_all-20261022-011024/raw/resources_launches.json).

### Occupancy across the v4 launch sweep

| Block | Blocks per SM | Occupancy % | Limited by | Fastest grid's median ms |
|---:|---:|---:|---|---:|
| 64 | 16 | 100.0 | registers, warp slots and the 16-block cap | 1.49 |
| 128 | 8 | 100.0 | registers and warp slots | 1.48 |
| 256 | 4 | 100.0 | registers and warp slots | 1.50 |
| 512 | 2 | 100.0 | registers and warp slots | 1.53 |
| 1,024 | 1 | 100.0 | registers and warp slots | 1.50 |

Every swept block size reaches the same theoretical occupancy, so occupancy does not explain the differences across the sweep; the grid size changes them more (see the launch configuration sweep above).

## Clock throttling

The driver's clock-event reasons were logged every 100 ms during the 12 blocks that time GPU work alone. In 5 of them the card reached its software power cap for part of the block (`ladder_kernel`, `ladder_kernel_queued`, `ladder_v0_alone`, `ladder_v4_sweep`, `slippage_full_pass`), so their figures were measured at a power-limited clock; none reported a thermal or hardware slowdown.
The 6 blocks that time host code or wall-clock transfers were not polled, so the poller could not disturb the host; for them the SM clock was sampled after the warmup and after the last iteration. None lost more than a tenth of its SM clock between the two, but 1 of them began below half the card's maximum clock, where that comparison cannot show a drop.

## Runs

- [`bench_all-20261022-011024`](results/bench_all-20261022-011024/manifest.json)
