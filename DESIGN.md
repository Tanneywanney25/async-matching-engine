# Design decisions

Rendered by `scripts/render_docs.py`; every number comes from a committed real-data T4 run.

## Offline GPU, never the live path

The matching engine answers in microseconds on one CPU core. A PCIe round trip to any discrete GPU costs more than that before a kernel even starts, so putting the card in the order path could only make it slower. The GPU instead replays the recorded history, where the work is wide (every snapshot, every order size) and latency does not matter. The pipeline measurements make the same point from the other side: transfers are 94.1% of stage time, so analyses run in one residency on the card.

## Integer ticks, not float32 prices

At the archive's highest top-of-book price, $84,856.07, a float32 has a spacing of 0.78 cents, 78% of the one-cent tick, so a float32 spread is mostly rounding error (`tests/test_archive.py` shows the failure on a real price). Prices are int32 tick counts; spreads and level indices are exact integers on the device, sizes stay float32, and the microprice is kept as an offset from the mid so float32 does not lose it.

## Scan precision

A rolling window formed as the difference of two prefix sums cancels: the prefix grows with the stream while the window does not. Over 1,726,306 top-of-book notionals the worst float32 window was off by 3.5e+00 of its own value, Kahan-compensated float32 by 1.3e-04, block-rebased float32 by 4.7e-03 and even float64 by 1.8e-08, above the budget of a tenth of a tick at the largest price (1.2e-08 relative). A double-double prefix (an unevaluated hi + lo pair built with TwoSum, exclusive prefixes shuffled rather than subtracted) is the only design inside it: its windows match the exact integer reference after rounding. It costs 3.11 ms against 0.406 ms for float32 per pass, negligible next to moving the data, so the rolling VWAP uses it; its one-second windows match `math.fsum` exactly where a float64 cumulative sum is off by up to 1.5e-05.

## Structure-of-arrays book levels

Level-major storage lets a warp read 32 consecutive snapshots of one level in one transaction. In Nsight Systems the same four kernels took 4,796 ms in array-of-structures order and 200 ms in structure-of-arrays order, a cut of 95.8%.

## Warp-cooperative slippage

A thread per (snapshot, side, size) walking the book diverges: lanes of one warp stop at different depths and the warp runs as long as its deepest order. One warp per snapshot instead builds the cumulative quantity and notional of all levels with shuffles and lets each lane binary-search two sizes, so every lane does the same work. Measured: 4.4 times faster, 77.1% of the naive kernel's time was divergence.

## What limits occupancy

The three kernels with the longest measured launches, from the resource table in BENCHMARKS.md: `slippage_naive` (286 ms per launch) uses 38 registers per thread and no shared memory, so at 256 threads per block 4 blocks fit on a multiprocessor, 100.0% occupancy, and warp slots bind first; registers would allow 6 blocks. `layout_cumdepth_aos` (111 ms per launch) uses 64 registers per thread and no shared memory, so at 256 threads per block 4 blocks fit on a multiprocessor, 100.0% occupancy, and registers and warp slots bind first. `slippage_warp` (65.4 ms per launch) uses 49 registers per thread and 10,240 bytes of static shared memory, so at 256 threads per block 4 blocks fit on a multiprocessor, 100.0% occupancy, and registers and warp slots bind first; shared memory would allow 6 blocks. All three already fill every warp slot the multiprocessor has, so more occupancy cannot speed them up; what does is what the other sections measured: the naive walk loses 77.1% of its time to divergence and the array-of-structures kernels read uncoalesced memory, and the same kernels in structure-of-arrays order take 95.8% less time. These are theoretical figures: without counters there is no achieved occupancy, so the table says what the compiler and launch shape allow, not what the scheduler reached.

## Nsight Systems rather than Nsight Compute

Kaggle's container denies access to the GPU performance counters (`ERR_NVGPUCTRPERM`), so Nsight Compute cannot run there (it was tried, see `DECISIONS.md`). Nsight Systems needs no counters and answers what this project asks: total kernel time per layout, stream overlap and transfer share. An SSH backend for a rented T4 with counters enabled is designed in `remote/README.md` for the day kernel-level metrics are needed.

## Honest baselines

The single-threaded C++ baseline is the one bullet 1 quotes, and it is compiled with `-O3 -march=native` and a cache-friendly level-outer loop. The same code with OpenMP on every core is reported too: the best kernel is 11.8 times faster than that, a smaller and fairer figure.
