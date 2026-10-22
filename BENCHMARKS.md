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
| Run | `bench_all-20261022-003544` at git `5fbb01146b54`, NVRTC target `sm_75` |
| Archive | 863,153 snapshots, 280,818 trades, source `real` |
| Capture window | `2026-10-21T07:49:24Z` to `2026-10-21T17:10:57Z` (9.36 hours), sequence gaps: 0 |
| Archive content hash | `10e3c7d79010feb6281d9c3833d00e6a05a51275dc9e1aa58c05c5aa75451959` |

## Memory bandwidth ceiling

| Measurement | Median ms (IQR) | GB/s (read + write) | Raw |
|---|---:|---:|---|
| Device-to-device copy kernel, 128-bit loads | 9.69 (IQR 0.0765) | 221.6 | [raw](results/bench_all-20261022-003544/raw/bandwidth__copy_kernel.json) |
| `cudaMemcpy` device to device | 9.21 (IQR 0.290) | 233.1 | [raw](results/bench_all-20261022-003544/raw/bandwidth__cudamemcpy_d2d.json) |
| Read-only streaming kernel, 128-bit loads (bytes read) | 4.09 (IQR 0.00906) | 262.8 | [raw](results/bench_all-20261022-003544/raw/bandwidth__read_kernel.json) |

A copy spends half its DRAM time writing and pays the read/write turnaround; the feature kernels read about seventeen bytes for every one they write, so the read-only figure is the fair ceiling for them and the highest measured figure is used.

Buffer size 1,073,741,824 bytes. Measured ceiling 262.8 GB/s, 82.1% of the 320.1 GB/s theoretical figure.

## Feature kernels: the optimization ladder

863,153 snapshots x 50 levels. Effective bandwidth counts the minimum traffic, 433 bytes per snapshot (every level's sizes, the top-of-book prices, the product id and six outputs). Per-kernel time is CUDA events around the launches only; end to end is wall clock including the host-to-device copy of exactly the inputs the kernels read and the copy back of the six outputs, from pinned host buffers, with the same from pageable NumPy memory beside it.

| Rung | Per-kernel median ms (IQR) | End to end ms | End to end, pageable ms | Effective GB/s | % of measured ceiling | % of theoretical | Block | Verified | Raw |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| v0 array-of-structures, one kernel per metric | 41.0 (IQR 24.6) | 100 | 193 | 9.1 | 3.5 | 2.8 | 256 | yes | [raw](results/bench_all-20261022-003544/raw/ladder_kernel__v0_aos.json) |
| v1 structure-of-arrays, one kernel per metric | 2.02 (IQR 0.0162) | 32.4 | 81.9 | 185.3 | 70.5 | 57.9 | 256 | yes | [raw](results/bench_all-20261022-003544/raw/ladder_kernel__v1_soa.json) |
| v2 structure-of-arrays, fused | 1.78 (IQR 0.0131) | 32.1 | 81.5 | 209.7 | 79.8 | 65.5 | 256 | yes | [raw](results/bench_all-20261022-003544/raw/ladder_kernel__v2_soa_fused.json) |
| v3 fused, 128-bit loads | 1.57 (IQR 0.00940) | 31.9 | 81.2 | 238.5 | 90.8 | 74.5 | 256 | yes | [raw](results/bench_all-20261022-003544/raw/ladder_kernel__v3_soa_fused_vec4.json) |
| v4 v3 + grid-stride, best launch configuration | 1.49 (IQR 0.0473) | 31.8 | 81.3 | 250.5 | 95.3 | 78.3 | 128 | yes | [raw](results/bench_all-20261022-003544/raw/ladder_kernel__v4_soa_fused_vec4_sweep.json) |
| Triton, fused like v2 (extra rung, best of a small sweep) | 1.75 (IQR 0.00842) | 32.1 | 81.5 | 213.9 | 81.4 | 66.8 | 1,024 | yes | [raw](results/bench_all-20261022-003544/raw/ladder_kernel__triton_fused.json) |

### Launch configuration sweep (v4, grid-stride)

| Block | Grid | Median ms | IQR ms | Effective GB/s |
|---:|---:|---:|---:|---:|
| 64 | 40 | 1.62 | 0.0186 | 230.9 |
| 64 | 80 | 1.53 | 0.0182 | 244.2 |
| 64 | 160 | 1.49 | 0.0163 | 250.9 |
| 64 | 320 | 1.64 | 0.0114 | 228.2 |
| 64 | 640 | 1.71 | 0.0124 | 218.0 |
| 64 | 1,280 | 1.71 | 0.0110 | 218.6 |
| 128 | 40 | 1.52 | 0.00758 | 245.9 |
| 128 | 80 | 1.49 | 0.0181 | 251.4 |
| 128 | 160 | 1.63 | 0.0144 | 229.8 |
| 128 | 320 | 1.69 | 0.0179 | 221.2 |
| 128 | 640 | 1.68 | 0.0154 | 222.1 |
| 128 | 1,280 | 1.65 | 0.00438 | 226.1 |
| 256 | 40 | 1.49 | 0.0121 | 250.4 |
| 256 | 80 | 1.64 | 0.0102 | 228.4 |
| 256 | 160 | 1.65 | 0.0108 | 226.4 |
| 256 | 320 | 1.63 | 0.0130 | 229.8 |
| 256 | 640 | 1.61 | 0.00834 | 231.7 |
| 256 | 1,280 | 1.60 | 0.00586 | 234.0 |
| 512 | 40 | 1.67 | 0.0129 | 224.2 |
| 512 | 80 | 1.61 | 0.0128 | 232.7 |
| 512 | 160 | 1.60 | 0.0179 | 233.6 |
| 512 | 320 | 1.56 | 0.0104 | 239.8 |
| 512 | 640 | 1.53 | 0.00926 | 244.5 |
| 512 | 1,280 | 1.52 | 0.00758 | 245.1 |
| 1,024 | 40 | 1.60 | 0.0196 | 232.9 |
| 1,024 | 80 | 1.55 | 0.0111 | 240.9 |
| 1,024 | 160 | 1.52 | 0.00865 | 245.1 |
| 1,024 | 320 | 1.50 | 0.00882 | 248.4 |
| 1,024 | 640 | 1.50 | 0.00786 | 248.6 |
| 1,024 | 1,280 | 1.50 | 0.00776 | 248.6 |

### GPU time with the launches queued ahead

The per-kernel times above record the start event on an idle GPU, so each includes the host's dispatch of the launch. This block, interleaved like the others, queues each pass behind a 1,000 µs busy-wait kernel, so the events bracket GPU execution only, and times how long the host took to enqueue the pass. The slowest enqueue took 127.11 µs, well inside the wait.

| Rung | Events around the launch, median ms | Launches queued, median ms (IQR) | Host enqueue, median µs |
|---|---:|---:|---:|
| v0 array-of-structures, one kernel per metric | 41.0 | 41.9 (IQR 0.228) | 50.97 |
| v1 structure-of-arrays, one kernel per metric | 2.02 | 2.00 (IQR 0.00782) | 58.64 |
| v2 structure-of-arrays, fused | 1.78 | 1.77 (IQR 0.00743) | 19.70 |
| v3 fused, 128-bit loads | 1.57 | 1.55 (IQR 0.00520) | 18.01 |
| v4 v3 + grid-stride, best launch configuration | 1.49 | 1.47 (IQR 0.00420) | 17.82 |
| Triton, fused like v2 (extra rung, best of a small sweep) | 1.75 | 1.69 (IQR 0.00814) | 67.57 |

### Extra rung: the fused kernel in Triton

Triton `3.6.0` compiles the same fused computation as v2 (level sums in level order, IEEE division) and reads the ladder's own device arrays, so it competes on identical data. Every launch configuration was checked against the NumPy reference before any was timed; the ones that passed, timed with launches queued ahead, GPU time only, in an interleaved block of their own: 256 snapshots and 4 warps per program: 1.81 ms; 512 snapshots and 4 warps per program: 1.75 ms; 1,024 snapshots and 4 warps per program: 1.76 ms; 1,024 snapshots and 8 warps per program: 1.69 ms. Measured with its launch queued ahead, the best one takes 0.95 times the GPU time of the CUDA v2 kernel it mirrors and 1.15 times that of the best CUDA rung (v4 v3 + grid-stride, best launch configuration). With the events around the launch, as in the ladder table, the ratios are 0.98 and 1.17: enqueuing one Triton launch took a median 67.57 µs on the host, against 19.70 µs for v2 through CuPy. It sits in the ladder table (its Block column is snapshots per program) and is never counted as the best rung. Its outputs are bit-for-bit identical to v2's.

## Baselines (same machine, same job, same archive)

| Implementation | Median ms (IQR) | Verified | Raw |
|---|---:|:---:|---|
| C++ single thread, `g++ -O3 -march=native` | 36.0 (IQR 0.595) | yes | [raw](results/bench_all-20261022-003544/raw/ladder_e2e_and_cpu__cpp_single.json) |
| C++ with OpenMP on every core | 17.3 (IQR 0.396) | yes | [raw](results/bench_all-20261022-003544/raw/ladder_e2e_and_cpu__cpp_omp.json) |
| Vectorized NumPy | 86.9 (IQR 1.37) | yes | [raw](results/bench_all-20261022-003544/raw/ladder_e2e_and_cpu__numpy.json) |
| CuPy expression chain, per kernel | 4.28 (IQR 1.72) | yes | [raw](results/bench_all-20261022-003544/raw/ladder_kernel__cupy_chain.json) |
| CuPy expression chain, end to end | 89.6 (IQR 0.713) | yes | [raw](results/bench_all-20261022-003544/raw/ladder_e2e_and_cpu__cupy_e2e.json) |

Best rung: `v4_soa_fused_vec4_sweep`. Per kernel it is 24.2 times faster than single-threaded C++, 11.6 times faster than C++ with OpenMP on every core and 2.9 times faster than the CuPy chain. End to end, transfers included, it is 2.7 times faster than NumPy and 1.1 times faster than the CuPy chain with its own transfers. From pageable memory the end-to-end advantage over NumPy shrinks to 1.1 times.

## Prefix scans

GB/s counts every float64 element twice, once read and once written; the last column compares that with the read-only streaming ceiling measured in the same job.

Before it was timed, the single-pass decoupled look-back scan (Merrill and Garland, 2016) passed 16 correctness cases in a child process with a timeout, at sizes from 1 to 16,777,216 elements: 8 of them on integer-valued inputs, whose partial sums are exact in float64, so they must equal `numpy.cumsum` bit for bit whatever order the look-back adds the tiles in, and 20 repeated runs at 16,777,216 elements with 0 mismatches.

### Archive-length series (863,153 float64 elements)

| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | % of read ceiling | Verified | Raw |
|---|---:|---:|---:|---:|:---:|---|
| blelloch_shared | 0.248 (IQR 0.00647) | 3.48 | 55.7 | 21.2 | yes | [raw](results/bench_all-20261022-003544/raw/scan_snapshots__blelloch_shared.json) |
| warp_shuffle | 0.184 (IQR 0.00722) | 4.68 | 74.9 | 28.5 | yes | [raw](results/bench_all-20261022-003544/raw/scan_snapshots__warp_shuffle.json) |
| decoupled_lookback | 0.145 (IQR 0.00710) | 5.96 | 95.4 | 36.3 | yes | [raw](results/bench_all-20261022-003544/raw/scan_snapshots__decoupled_lookback.json) |
| cub_device_scan | 0.119 (IQR 0.00572) | 7.27 | 116.3 | 44.2 | yes | [raw](results/bench_all-20261022-003544/raw/scan_snapshots__cub_device_scan.json) |

Winner: `cub_device_scan`. The decoupled look-back scan ran in 0.145 ms against CUB's 0.119 ms: CUB is still faster, and the hand-written single pass is 21.9% slower.

### Large series (16,777,216 float64 elements)

| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | % of read ceiling | Verified | Raw |
|---|---:|---:|---:|---:|:---:|---|
| blelloch_shared | 3.78 (IQR 0.00763) | 4.44 | 71.0 | 27.0 | yes | [raw](results/bench_all-20261022-003544/raw/scan_large__blelloch_shared.json) |
| warp_shuffle | 2.66 (IQR 0.00810) | 6.30 | 100.8 | 38.4 | yes | [raw](results/bench_all-20261022-003544/raw/scan_large__warp_shuffle.json) |
| decoupled_lookback | 1.68 (IQR 0.0158) | 9.98 | 159.7 | 60.8 | yes | [raw](results/bench_all-20261022-003544/raw/scan_large__decoupled_lookback.json) |
| cub_device_scan | 1.61 (IQR 0.00763) | 10.40 | 166.4 | 63.3 | yes | [raw](results/bench_all-20261022-003544/raw/scan_large__cub_device_scan.json) |

Winner: `cub_device_scan`. The decoupled look-back scan ran in 1.68 ms against CUB's 1.61 ms: CUB is still faster, and the hand-written single pass is 4.2% slower.

### Rolling windows

- Rolling volume-weighted average price over 280,818 trades (100-trade window, and time windows sampled at 863,153 snapshots): 1.00 (IQR 0.00790) ms for the whole chain; maximum relative error against `metrics.rolling_vwap` in Decimal 1.9e-16 over 200 sampled trades. Prefixes are double-double; against `math.fsum` window sums at 2,000 sampled snapshots the one-second window's maximum relative error is 0.0e+00 (a plain float64 cumulative sum reaches 1.5e-05).
- Rolling one-second mean spread and imbalance over 863,153 snapshots: 1.20 (IQR 0.0117) ms, 716.7 million snapshots per second.

## Prefix-sum precision

Window sums of 100 values over 1,726,306 float32 top-of-book notionals, against an exact integer reference.

| Variant | Max relative error | Median relative error | Median ms (IQR) |
|---|---:|---:|---:|
| float32_naive | 3.5e+00 | 2.5e-04 | 0.424 (IQR 0.00399) |
| float64 | 1.8e-08 | 4.1e-15 | 0.448 (IQR 0.00387) |
| kahan_float32 | 1.3e-04 | 1.0e-08 | 1.06 (IQR 0.00692) |
| block_rebased_float32 | 4.7e-03 | 2.1e-07 | 0.516 (IQR 0.00130) |
| double_double (production) | 0.0e+00 | 0.0e+00 | 3.60 (IQR 0.00267) |

Production choice `double_double` costs 748.7% time against float32 naive.

## Layout under Nsight Systems

Total GPU time of the four book-level kernels over 30 timed passes plus the same warmup passes in both layouts, from `nsys stats --report cuda_gpu_kern_sum`.

| Kernel | Array-of-structures ms | Structure-of-arrays ms | Reduction % |
|---|---:|---:|---:|
| cumdepth | 3,673 | 144 | 96.1 |
| depth | 958 | 49.0 | 94.9 |
| imbalance | 190 | 9.85 | 94.8 |
| spread | 22.7 | 1.45 | 93.6 |
| **Total** | 4,843 | 204 | **95.8** |

Reports: [aos.nsys-rep](results/bench_all-20261022-003544/raw/profiles/aos.nsys-rep), [soa.nsys-rep](results/bench_all-20261022-003544/raw/profiles/soa.nsys-rep), [aos CSV](results/bench_all-20261022-003544/raw/profiles/aos_kern_sum.csv), [soa CSV](results/bench_all-20261022-003544/raw/profiles/soa_kern_sum.csv). Nsight Compute: blocked (`==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Perfor`).

## Execution-cost surface

110,483,584 simulated orders: 863,153 snapshots x both sides x 64 order sizes.

| Kernel | Median ms (IQR) | Orders per second (billions) | Raw |
|---|---:|---:|---|
| Naive, one thread per order | 282 (IQR 2.48) | 0.39 | [raw](results/bench_all-20261022-003544/raw/slippage_full_pass__naive.json) |
| Warp-cooperative | 62.4 (IQR 1.65) | 1.77 | [raw](results/bench_all-20261022-003544/raw/slippage_full_pass__warp.json) |

Warp-cooperative speedup 4.5 times; divergence penalty of the naive walk 77.9%. Verified against the Decimal walker on 1,000 snapshots: exact ticks `True`, maximum average-price error 2.1e-04 ticks. Surface: [per-minute medians](results/bench_all-20261022-003544/raw/slippage_surface.json).

## Order flow imbalance

10-level order flow imbalance (Cont, Kukanov and Stoikov), stencil kernel 2.08 (IQR 0.0781) ms. Out-of-sample fit on the last 30% of each product's time series:

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
| Overlapped pass (wall clock) | 57.6 (IQR 0.0321) ms |
| Serial pass, one stream | 61.9 (IQR 0.0420) ms |
| Saved by overlap | 7.0% |
| Host to device | 57.2 ms at 12.1 GB/s |
| Kernels | 3.64 ms |
| Device to host | 1.69 ms at 10.2 GB/s |
| Transfer share of stage time | 94.2% |

Timeline: [SVG from the Nsight Systems trace](results/bench_all-20261022-003544/raw/pipeline_timeline.svg).

### Narrower prices on the wire: int16 tick offsets

Each side's best price travels as int32 per snapshot and every level as an int16 offset from it, decoded inside the kernel; a chunk holding any offset beyond 32,767 ticks is sent as int32 instead.

| Measurement | int32 | int16 offsets |
|---|---:|---:|
| Bytes host to device per pass | 690,536,448 | 524,812,288 |
| End to end, overlapped (median ms, IQR) | 57.6 (IQR 0.0393) | 43.2 (IQR 0.0460) |
| Host to device, summed over chunks (ms) | 57.2 | 42.8 |

The encoding moves 24.0% fewer bytes; none of the 16 chunks needed the int32 fallback, because every recorded level sat within 32,767 ticks of its side's best price. End to end the pass is 25.0% faster, and its outputs are bit-for-bit identical to the int32 pass's (verified against the NumPy reference too).

### One card against two

`nvidia-smi topo -m` reports the two cards as `PHB`, connected through a PCIe host bridge, that is the CPU; each has its own PCIe link (`GPU0` at `00000000:00:04.0`, rated `Gen3 x16` and running at `Gen3 x16` just after the copies, `GPU1` at `00000000:00:05.0`, rated `Gen3 x16` and running at `Gen3 x16` just after the copies). The archive is split in two contiguous halves of 8 chunks, each card with its own pinned host buffer, module, streams and device buffers, and the two cards' chunks are issued alternately.

| Measurement | One card | Two cards |
|---|---:|---:|
| Whole pipeline, overlapped (median ms, IQR) | 57.6 (IQR 0.0405) | 28.6 (IQR 0.0275) |
| Copy-only pass, host to device (median ms, IQR) | 57.0 (IQR 0.0236) | 28.1 (IQR 0.0175) |
| Host-to-device bandwidth (GB/s, aggregate for two) | 12.1 | 24.6 |

Two cards take 28.6 ms against 57.6 ms on one, a speedup of 2.02 (100.8% of perfect scaling), and copy 2.03 times as many bytes per second as one card: the two cards' copies run side by side at close to each card's own rate, so each T4 has its own path to host memory, and the whole pipeline scales with the second card. A single-threaded host memcpy on this machine runs at 5.5 GB/s for comparison. The two-card outputs are bit-for-bit identical to the one-card outputs and match the NumPy reference. An earlier full run, `bench_rerun-20261021-233009`, on another Kaggle host that `nvidia-smi` described the same way, found that two cards copied no faster than one; DECISIONS.md compares the two. This page reports the run above.

## Matching engine (CPU)

`scripts/benchmark.py`, 100,000 orders and 10,000 cancels, run on the T4 job's host CPU (`Intel(R) Xeon(R) CPU @ 2.00GHz`, Python `3.13.15`). Matching path only: an in-memory book, no feed, no network.

| Operation | Mean µs | p50 µs | p95 µs | p99 µs |
|---|---:|---:|---:|---:|
| `add_order` | 10.35 | 7.68 | 22.44 | 31.82 |
| `cancel_order` | 1.82 | 0.36 | 5.12 | 6.04 |

Raw latencies: [raw](results/bench_all-20261022-003544/raw/engine_latency_us.json). The laptop's own figures are in the README's Benchmark table.

## C++20 order book core (CPU)

`native/`, built with CMake in the same Kaggle job (`c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, -O3 (CMake Release), C++20) and run on its host CPU (`Intel(R) Xeon(R) CPU @ 2.00GHz`). Its native tests passed 2,482,227 checks, among them a randomized differential test against a `std::map` and `std::deque` reference book, and a counting allocator saw 0 heap allocations over 200,000 operations once the book was built. The differential fuzzer ran 10,000 seeded streams (548,317 operations) against the reference book with 0 divergences, comparing fills, statuses, best prices and depth after every operation.

Stream: scripts/benchmark.py's stream: 100,000 limit orders, price gauss(mid, 1% of mid) to the cent, size uniform(0.001, 2.0) to 1e-6, seed 42; 10,000 cancels of rested ids. Every path below replays the same operations in one process; they agreed on fills, resting orders and cancels found: yes.

| Path | add p50 µs | add p99 µs | add p99.9 µs | cancel p50 µs | cancel p99 µs | cancel p99.9 µs |
|---|---:|---:|---:|---:|---:|---:|
| C++ core, native harness (no Python in the loop) | 0.082 | 0.400 | 0.596 | 0.051 | 0.418 | 0.571 |
| C++ core through the pybind11 binding, integer ticks and lots | 0.518 | 0.948 | 6.686 | 0.397 | 0.932 | 2.673 |
| C++ core through `NativeOrderBook`, Decimal orders | 3.473 | 15.5 | 28.5 | 0.527 | 1.685 | 4.596 |
| Python engine (the reference implementation) | 6.316 | 29.1 | 54.3 | 0.436 | 7.531 | 20.8 |

Of the 10,000 cancels, 3,575 found a resting order; the others name orders the stream had already filled, so their latency is a lookup that finds nothing. Every ratio here uses the Python engine row of this table, measured in the same process on the same stream (`scripts/benchmark.py`, run separately in this job, reported a p99 `add_order` of 31.82 µs). At p99 the native `add` is 72.8 times faster than the Python engine's, 30.7 times faster through the binding with integers and 1.9 times faster with Decimal orders. The binding rows differ from the native one only by the call from Python, and from each other only by the Decimal conversion in `NativeOrderBook`. The native harness reads `std::chrono::steady_clock` twice per operation; a pair of reads costs 0.024 µs at the median, included above, not subtracted. Raw: [raw](results/bench_all-20261022-003544/raw/synthetic_native_engine_latency_us.json).

### The same paths on a stream derived from the recorded book

Stream: derived from the first 95.5 minutes of the recorded BTC-USD book: 82,059 snapshots reconciled into limit orders and cancels, 26,311 trades replayed as market orders on the aggressor's side (1,000,020 operations: 491,040 limit orders, 26,311 market orders, 482,669 cancels). Removals come before additions within each snapshot, and no limit order matched on arrival, so every fill comes from a market order. The paths agreed on fills, resting orders and cancels found: yes.

| Path | limit p50 µs | limit p99 µs | market p50 µs | market p99 µs | cancel p50 µs | cancel p99 µs |
|---|---:|---:|---:|---:|---:|---:|
| C++ core, native harness (no Python in the loop) | 0.061 | 0.147 | 0.039 | 0.204 | 0.050 | 0.104 |
| C++ core through the pybind11 binding, integer ticks and lots | 0.547 | 0.937 | 0.603 | 3.322 | 0.383 | 0.694 |
| C++ core through `NativeOrderBook`, Decimal orders | 2.908 | 6.229 | 4.354 | 17.5 | 0.571 | 1.006 |
| Python engine (the reference implementation) | 5.880 | 14.5 | 6.569 | 30.1 | 2.434 | 6.324 |

Against the Python engine row, measured in the same process on the same operations, the native core is 98.4 times faster at p99 on limit orders and 147 times faster on market orders; on cancels it is 60.8 times faster. On the synthetic stream above the `add` ratio is 72.8. A pair of clock reads in this mode costs 0.024 µs at the median, included above. Raw: [raw](results/bench_all-20261022-003544/raw/realistic_native_engine_latency_us.json).

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

CUB's kernels are compiled into `libscan_cub.so`, so their registers and shared memory come from `cuobjdump --dump-resource-usage` (which reports the stack frame rather than spills) and their launch shape and blocks per SM from CUB's own `-DCUB_DEBUG_LOG` output, which computes occupancy with the same API. Raw: [ptxas logs and CUB log](results/bench_all-20261022-003544/raw/resources), [launch configurations](results/bench_all-20261022-003544/raw/resources_launches.json).

### Occupancy across the v4 launch sweep

| Block | Blocks per SM | Occupancy % | Limited by | Fastest grid's median ms |
|---:|---:|---:|---|---:|
| 64 | 16 | 100.0 | registers, warp slots and the 16-block cap | 1.49 |
| 128 | 8 | 100.0 | registers and warp slots | 1.49 |
| 256 | 4 | 100.0 | registers and warp slots | 1.49 |
| 512 | 2 | 100.0 | registers and warp slots | 1.52 |
| 1,024 | 1 | 100.0 | registers and warp slots | 1.50 |

Every swept block size reaches the same theoretical occupancy, so occupancy does not explain the differences across the sweep; the grid size changes them more (see the launch configuration sweep above).

## Clock throttling

No timed block lost more than a tenth of its SM clock between the samples taken before and after it, on every card the block used.

## Runs

- [`bench_all-20261022-003544`](results/bench_all-20261022-003544/manifest.json)
