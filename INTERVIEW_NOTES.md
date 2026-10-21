# Interview notes

Rendered from the measured numbers by `scripts/render_docs.py`.

## Questions

### 1. Why is the end-to-end speedup so much smaller than the per-kernel one?

The kernel is 25.8 times faster than single-threaded C++, but end to end it is 3.2 times faster than NumPy because the inputs cross PCIe first. The work is a few adds per byte, so the copy dominates; that is why analyses run in one residency.

### 2. How close to the hardware limit is the best feature kernel?

251.1 GB/s of minimum traffic, 95.6% of the 262.7 GB/s the read-only streaming kernel achieves (a copy is slower because half its traffic is writes) and 78.5% of the theoretical figure.

### 3. What did each rung of the ladder buy?

Array-of-structures to structure-of-arrays fixes coalescing, fusion removes repeated passes over the same sizes, 128-bit loads cut instruction count, and the sweep picks the launch shape; the table in BENCHMARKS.md has each rung's median.

### 4. Is the arithmetic intensity high enough for the GPU to be compute bound?

No: 129 floating-point operations per 433 bytes, far left of the roofline's ridge point, so memory bandwidth is the only ceiling that matters.

### 5. Why compare against OpenMP when the bullet quotes single-threaded C++?

Because an honest comparison beats a flattering one: with every core the CPU closes the gap to 12.4 times.

### 6. How did you measure the layout effect?

The same four kernels in both layouts under `nsys profile`, total kernel time from `cuda_gpu_kern_sum`: 4,782 ms versus 202 ms, 95.8% less.

### 7. Which scan won?

`cub_device_scan` on 16,777,216 elements. CUB's single-pass decoupled look-back reads and writes each element once; the hand-written scans need a second pass for the block carries.

### 8. How do you know the rolling price is right?

Against `metrics.rolling_vwap` in Decimal on sampled trades: maximum relative error 1.9e-16.

### 9. Why not float32, or even float64, for the prefix sums?

Window error reached 3.5e+00 relative in float32 and 1.8e-08 in float64, both above a tenth of a tick; the double-double scan stays inside it at 744.8% extra time over float32, which is still a few milliseconds.

### 10. What is divergence and how big was it here?

Lanes of a warp taking different branch paths serialize. The thread-per-order walk lost 76.9% of its time to it; the warp-cooperative kernel runs 4.3 times faster.

### 11. How was the execution-cost kernel verified?

On 1,000 random snapshots against a Decimal book walker, exact in ticks for the levels consumed.

### 12. Did the three streams actually overlap?

Yes, the Nsight Systems timeline shows copy and compute lanes active together; overlap saved 6.9% of the serial wall time, and transfers were still 94.2% of stage time.

### 13. Does order flow imbalance predict anything?

Not the future mid at these horizons: out of sample the fit is no better than the mean (R squared -0.0020 for BTC-USD at one second, -0.0099 at ten), which is what a liquid market should show; the study is a description, not a strategy.

### 14. Why is the matching engine in Python, and how fast is it?

Clarity and Decimal exactness. On the job's host CPU `add_order` p99 was 32.21 microseconds for the matching path alone; the GPU would add a PCIe round trip on top.

### 15. Why integer ticks?

At the archive's highest price, $84,856.07, neighbouring float32 values are 0.78 cents apart, so a one-cent spread cannot be represented exactly; int32 ticks make every price comparison exact.

## Three things that went wrong

- The first full real-data run put the best feature kernel at `105.8`% of the "measured ceiling", which was a copy kernel: a copy spends half its DRAM time writing, while the feature kernels read about seventeen bytes per byte written. Added a read-only `128`-bit streaming kernel to the bandwidth section and took the highest measured figure as the ceiling (rerun `bench_rerun-20261021-191923`); the best rung is now below it.
- Float64 was not precise enough for rolling windows built from prefix sums: on the real archive the precision study's float64 window error (`1.8e-8`) exceeded a tenth of a tick, and one-second VWAP windows holding only dust trades disagreed with a NumPy reference by `9e-5`. Fixed with a double-double prefix scan (`cuda/ddscan.cu`), now the production design for the VWAP chain, and with an exact `math.fsum` check instead of a float64 reference that had the same flaw.
- Three T4 smoke runs were spent on one class of bug: NVRTC rejects options given twice, and CuPy's `RawModule` already passes `-arch` and `-default-device`, so every kernel failed to compile until the harness stopped passing them. The fix was a one-line change; the lesson was to run a cheap quick job before every full one, which then caught the stale-export `nsys stats` failure the same way.
