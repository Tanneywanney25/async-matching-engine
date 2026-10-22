# Interview notes

Rendered from the measured numbers by `scripts/render_docs.py`.

## Questions

### 1. Why is the end-to-end speedup so much smaller than the per-kernel one?

The kernel is 23.5 times faster than single-threaded C++, but end to end it is 2.7 times faster than NumPy because the inputs cross PCIe first. The work is a few adds per byte, so the copy dominates; that is why analyses run in one residency.

### 2. How close to the hardware limit is the best feature kernel?

251.1 GB/s of minimum traffic, 95.7% of the 262.3 GB/s the read-only streaming kernel achieves (a copy is slower because half its traffic is writes) and 78.5% of the theoretical figure.

### 3. What did each rung of the ladder buy?

Array-of-structures to structure-of-arrays fixes coalescing, fusion removes repeated passes over the same sizes, 128-bit loads cut instruction count, and the sweep picks the launch shape; the table in BENCHMARKS.md has each rung's median.

### 4. Is the arithmetic intensity high enough for the GPU to be compute bound?

No: 129 floating-point operations per 433 bytes, far left of the roofline's ridge point, so memory bandwidth is the only ceiling that matters.

### 5. Why compare against OpenMP when the bullet quotes single-threaded C++?

Because an honest comparison beats a flattering one: with every core the CPU closes the gap to 11.5 times.

### 6. How did you measure the layout effect?

The same four kernels in both layouts under `nsys profile`, total kernel time from `cuda_gpu_kern_sum`: 4,848 ms versus 204 ms, 95.8% less.

### 7. Which scan won?

`cub_device_scan` on 16,777,216 elements. The hand-written single-pass decoupled look-back scan came within 3.5% of it (1.79 ms against 1.73 ms); both read and write each element once, while the Blelloch and warp-shuffle scans need a second pass for the block carries.

### 8. How do you know the rolling price is right?

Against `metrics.rolling_vwap` in Decimal on sampled trades: maximum relative error 1.9e-16.

### 9. Why not float32, or even float64, for the prefix sums?

Window error reached 3.5e+00 relative in float32 and 1.8e-08 in float64, both above a tenth of a tick; the double-double scan stays inside it at 645.4% extra time over float32, which is still a few milliseconds.

### 10. What is divergence and how big was it here?

Lanes of a warp taking different branch paths serialize. The thread-per-order walk lost 78.0% of its time to it; the warp-cooperative kernel runs 4.6 times faster.

### 11. How was the execution-cost kernel verified?

On 1,000 random snapshots against a Decimal book walker, exact in ticks for the levels consumed.

### 12. Did the three streams actually overlap?

Yes, the Nsight Systems timeline shows copy and compute lanes active together; overlap saved 6.9% of the serial wall time, and transfers were still 94.2% of stage time.

### 13. Does order flow imbalance predict anything?

Not the future mid at these horizons: out of sample the fit is no better than the mean (R squared -0.0020 for BTC-USD at one second, -0.0099 at ten), which is what a liquid market should show; the study is a description, not a strategy.

### 14. Why is the matching engine in Python, and how fast is it?

Clarity and Decimal exactness. On the job's host CPU `add_order` p99 was 31.82 microseconds for the matching path alone; the GPU would add a PCIe round trip on top.

### 15. Without hardware counters, how do you know what limits a kernel?

From the compiler and the occupancy API: `nvcc -Xptxas -v` gives registers, spills and shared memory, and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` gives resident blocks at the benchmarked block size. 43 of 46 launched kernels reach full theoretical occupancy, the three slowest among them, so for those the answer lies elsewhere: divergence and uncoalesced reads, which the slippage and layout sections measure. It is a theoretical figure, not achieved occupancy.

### 16. How does the single-pass scan avoid deadlock, and how did you test it?

Blocks take tile numbers from an atomic counter instead of trusting the launch order, so every tile a block waits on belongs to a block that is already running; statuses are published with release stores and read with acquire loads. Before timing, it ran in a child process with a timeout on integer inputs that must match `numpy.cumsum` bit for bit at any summation order, 20 times at 16,777,216 elements with 0 mismatches.

### 17. Why did the second T4 not speed up the pipeline?

Because the pipeline is transfer-bound and, in this session, the copies hit a limit both cards shared: the copy-only pass moved 12.1 GB/s on one card and 12.3 GB/s on two together, and `nvidia-smi topo -m` reports their connection as `PHB`, through a PCIe host bridge, that is the CPU. Here the second card added only compute, which this pipeline barely uses; sending fewer bytes helps either way. That holds in this session, while 1 of the 2 other full runs, each in its own Kaggle session reporting the same topology, showed close to double, so the result varies between sessions, and the runs do not show why.

### 18. What did narrowing prices to int16 buy?

24.0% fewer bytes over PCIe and 43.2 ms instead of 57.6 ms end to end, with outputs identical bit for bit, because each level is sent as an offset from its side's best price and 0 chunks needed the int32 fallback.

### 19. How do you know the C++ core has the Python engine's semantics?

The differential fuzzer checks it against the naive list-based reference book that the test suite also checks the Python engine against (`tests/test_differential.py`): 10,000 seeded streams, 548,317 operations, 0 divergences, comparing fills, statuses, best prices and depth after every operation. The benchmark also checks that all four replay paths, the Python engine among them, agree on fills, resting orders and cancels found.

### 20. Where does the time go when the C++ core is called from Python?

Natively an `add` costs 0.077 microseconds at the median; through the binding with integers 0.508, and with Decimal orders 3.428, so the Python call costs more than the matching itself, and the Decimal path adds the conversion and the Python fill objects on top. Decimal is converted only at that boundary, so the core works in integer ticks.

### 21. Why integer ticks?

At the archive's highest price, $84,856.07, neighbouring float32 values are 0.78 cents apart, so a one-cent spread cannot be represented exactly; int32 ticks make every price comparison exact.

## Three things that went wrong

- The first phase never sampled a clock: Kaggle's T4 sessions have two cards, so `nvidia-smi --query-gpu` prints two lines and `bench.common.gpu_state()` (which unpacked exactly five comma-separated values) raised on every call. Every raw file in `bench_all-20261021-190618` and `bench_rerun-20261021-191923` records `ValueError('too many values to unpack (expected 5)')` instead of clocks, so the published "no timed block lost more than a tenth of its SM clock" was vacuous, and the bootstrap's manifests carried values like `'5001\nTesla T4'` for the second field onwards. `gpu_state()` now queries by card index, blocks sample the cards they use (`devices=`), a drop on any of them flags the block, a block without samples records `clocks_missing`, and the renderer must say so rather than claim a clean run. The phase `1` directories stay as they are; the final phase `2` run replaces their numbers.
- Three T4 smoke runs were spent on one class of bug: NVRTC rejects options given twice, and CuPy's `RawModule` already passes `-arch` and `-default-device`, so every kernel failed to compile until the harness stopped passing them. The fix was a one-line change; the lesson was to run a cheap quick job before every full one, which then caught the stale-export `nsys stats` failure the same way.
- Float64 was not precise enough for rolling windows built from prefix sums: on the real archive the precision study's float64 window error (`1.8e-8`) exceeded a tenth of a tick, and one-second VWAP windows holding only dust trades disagreed with a NumPy reference by `9e-5`. Fixed with a double-double prefix scan (`cuda/ddscan.cu`), now the production design for the VWAP chain, and with an exact `math.fsum` check instead of a float64 reference that had the same flaw.
