# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-10-21

### Added
- Kernel resource table: registers, spills and shared memory from `nvcc -Xptxas -v`, theoretical occupancy from the
  occupancy API at every benchmarked launch configuration (recorded as the benchmarks run), and the binding resource.
- Single-pass decoupled look-back scan (Merrill and Garland, 2016) with an atomic tile counter and release/acquire status
  words, checked in a child process with a timeout before it is timed against Blelloch, warp shuffle and CUB.
- int16 tick-offset encoding measured on the real archive, and the pipeline split across both T4s with its own streams and
  pinned buffers per card; outputs are compared bit for bit with the one-card, int32 pass.
- C++20 order book core (`native/`, CMake): tick-indexed price window with an ordered fallback, intrusive FIFO levels,
  pooled orders, open-addressing id map, no heap allocation on the hot path; pybind11 binding, `NativeOrderBook`,
  `fuzz_engine.py --engine native`, a native latency harness and `benchmark_native.py`.
- GPU Lab panels for kernel resources, transfers and scaling, and the C++ core's latency; new interview questions and bullets.
- CI builds the C++ core, runs its native tests, fuzzes it against the reference and compile-checks every CUDA kernel.

### Fixed
- A raw template expression in `INTERVIEW_NOTES.md`; the renderer now refuses to write unrendered template syntax.
- `check_numbers.py` matched numbers across documents; it now requires provenance in the same document and covers
  `DESIGN.md` and `INTERVIEW_NOTES.md`.
- GPU clocks were never sampled on Kaggle's two-card sessions; every block now samples the cards it uses.
- Remote jobs ran a pinned branch; they now run the pushed SHA of the current branch under a phase budget.

## [0.2.0] - 2026-10-21

### Added
- Raw Coinbase recorder (`recorder.py`, `scripts/record.py`) and a deterministic tick-encoded level-major snapshot archive.
- Kaggle T4 runner (`remote/`), benchmark harness (`bench/`) and CUDA kernels (`cuda/`): feature ladder, scans, precision study,
  layout profiling, execution-cost surface, order flow imbalance and a three-stream pipeline.
- Hypothesis invariants and a differential fuzzer for the matching engine.
- GPU Lab dashboard section, `/analytics` routes, documentation rendered from committed results.

## [0.1.0] - 2026-09-21

### Added
- `models` — `Side`, `OrderType`, `OrderStatus` enums and `Order`, `Fill`,
  `BookLevel`, `Trade` dataclasses, all using `decimal.Decimal`.
- `orderbook` — price-time-priority limit order book on `SortedDict`/`deque`
  with `add_order`, `cancel_order`, `match_market_order`, and depth/imbalance
  views; O(log n) operations.
- `feed` — `CoinbaseFeed` async consumer of the public Coinbase Advanced Trade
  WebSocket (level2, market_trades, heartbeats) with a book mirror, trade
  buffer, heartbeat gap detection, and exponential-backoff reconnection.
- `metrics` — rolling VWAP, book imbalance, and trade-flow imbalance.
- `engine` — `MatchingEngine` simulator with P&L accounting, a FastAPI service
  (`/order`, `/book`, `/trades`, `/metrics`), and a stdin CLI.
- `dashboard` — Rich terminal UI with a live order book, metrics, and trade tape.
- `scripts/benchmark.py` — 100K-order latency benchmark (p50/p95/p99).
- `scripts/replay.py` — offline synthetic feed replay.
- Static Vercel web dashboard (`public/`) and serverless order simulator (`api/`).
- Test suite covering the order book, metrics, engine, feed, and models.
