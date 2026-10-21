# Resume bullets

Rendered by `scripts/render_docs.py` from real-data T4 results; `scripts/check_bullets.py` checks the two-line length rule and the style rules. Sources: [BENCHMARKS.md](BENCHMARKS.md).

## Template bullet 1 (feature kernels)

<!-- bullet:1 -->
CUDA C++ kernels on an NVIDIA T4 for spread, depth and order imbalance over 863,153 recorded Coinbase book
snapshots, 23.5 times faster than single-threaded C++ per kernel and 2.8 times faster than NumPy end to end.

## Template bullet 2 (prefix scan and layout)

<!-- bullet:2 -->
Shared-memory parallel prefix scan for rolling volume-weighted average price with book levels stored in
structure-of-arrays order for coalesced reads, cutting measured kernel time 95.8% in Nsight Systems profiles.

## Template bullet 3 (matching engine)

<!-- bullet:3 -->
Asynchronous Python matching engine with price-time priority, partial fills and cancellations on Coinbase's
live WebSocket order book feed, benchmarked over 100,000 orders at 39.35 microseconds 99th-percentile latency.

## Alternate: GPU performance (percentage of peak bandwidth)

<!-- bullet:4 -->
Took CUDA order book feature kernels on an NVIDIA T4 from 9.1 to 251.3 gigabytes per second of effective
bandwidth, 95.8% of the measured memory ceiling, by moving book levels to structure-of-arrays order.

## Alternate: execution cost (the slippage kernel)

<!-- bullet:5 -->
Wrote a warp-cooperative CUDA kernel that prices orders of 64 sizes against every recorded book snapshot
at 1.69 billion simulated orders per second, 4.4 times faster than a divergent thread-per-order walk.
